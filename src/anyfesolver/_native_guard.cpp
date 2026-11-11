// Workspace candidate only. CPython 3.11 full API; no numerical operations.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cstdint>
#include <cstring>
#include <new>
#include <algorithm>
#include <memory>
#include <utility>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#ifndef ANYSOLVER_GUARD_SOURCE_SHA
#define ANYSOLVER_GUARD_SOURCE_SHA "workspace-unbuilt"
#endif

namespace {
enum class Result { Equal, Changed, Unsupported, Error };
struct Statistics {
    std::uint64_t calls=0,passes=0,mismatches=0,unsupported=0,owner_changes=0,errors=0;
    std::uint64_t classes=0,helpers=0,callable_nodes=0,authority_nodes=0,global_bindings=0;
};
Statistics statistics;
struct Ref {
    PyObject* p = nullptr;
    Ref() = default;
    explicit Ref(PyObject* value, bool borrowed=false):p(value) {if(borrowed)Py_XINCREF(p);}
    Ref(const Ref& other):p(other.p){Py_XINCREF(p);}
    Ref(Ref&& other) noexcept:p(other.p){other.p=nullptr;}
    Ref& operator=(Ref other) noexcept{std::swap(p,other.p);return *this;}
    ~Ref(){Py_XDECREF(p);}
    explicit operator bool()const{return p!=nullptr;}
};
bool tuple(PyObject* value, Py_ssize_t size=-1) {
    return PyTuple_CheckExact(value)&&(size<0||PyTuple_GET_SIZE(value)==size);
}
bool text(PyObject* value){return PyUnicode_CheckExact(value);}
Result string_equal(PyObject* value,const char* expected) {
    if(!text(value))return Result::Unsupported;
    int same=PyUnicode_CompareWithASCIIString(value,expected);
    if(same==-1&&PyErr_Occurred())return Result::Error;
    return same==0?Result::Equal:Result::Changed;
}
Result object_id(PyObject* expected,PyObject* actual) {
    if(!PyLong_CheckExact(expected))return Result::Unsupported;
    void* pointer=PyLong_AsVoidPtr(expected);
    if(PyErr_Occurred()) {
        if(PyErr_ExceptionMatches(PyExc_OverflowError)){PyErr_Clear();return Result::Changed;}
        return Result::Error;
    }
    return pointer==actual?Result::Equal:Result::Changed;
}
bool exact_function(PyObject* object){return Py_IS_TYPE(object,&PyFunction_Type);}
bool namespace_lookup(PyObject* ns,const char* name,PyObject*& found) {
    if(!ns||!PyDict_CheckExact(ns))return false;
    found=nullptr;Py_ssize_t p=0;PyObject *key,*value;
    // Validate every key before a dict lookup can compare a subclass key.
    while(PyDict_Next(ns,&p,&key,&value))if(!text(key))return false;
    p=0;while(PyDict_Next(ns,&p,&key,&value)) {
        int same=PyUnicode_CompareWithASCIIString(key,name);
        if(same==-1&&PyErr_Occurred())return false;
        if(same==0){found=value;break;}
    }
    return true;
}

// Only generic type access with builtin MRO/dict descriptors is supported.
// Unknown metaclass routing declines before invoking Python equality/getattr.
bool type_access_supported(PyObject* cls) {
    if(!PyType_Check(cls))return false;
    PyTypeObject* meta=Py_TYPE(cls);
    if(meta->tp_getattro!=PyType_Type.tp_getattro||meta->tp_getattr!=nullptr)return false;
    if(!meta->tp_mro||!tuple(meta->tp_mro))return false;
    for(const char* name: {"__mro__","__dict__"}) {
        PyObject* stock=PyDict_GetItemString(PyType_Type.tp_dict,name);
        PyObject* resolved=nullptr;
        for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(meta->tp_mro);++i) {
            PyObject* base=PyTuple_GET_ITEM(meta->tp_mro,i);
            if(!PyType_Check(base))return false;
            PyObject* ns=reinterpret_cast<PyTypeObject*>(base)->tp_dict;
            if(!namespace_lookup(ns,name,resolved))return false;
            if(resolved)break;
        }
        if(resolved!=stock)return false;
    }
    return true;
}

struct ClassRecord {Ref cls,expected;};
struct EnvironmentRecord {
    Ref function,code,globals,builtins;
    Ref defaults,keywords,closure;
    std::vector<Ref> default_values,keyword_pairs,cell_values;
    // Exact dict/name/presence/value tuples; no getattr or Python callbacks.
    std::vector<Ref> bindings;
};
struct GuardPlan {
    Ref classes,helpers,missing;
    std::vector<ClassRecord> records;
    std::vector<EnvironmentRecord> environments;
    // Cold stock-code grammar only; no mutable function or namespace proof.
    std::unordered_map<PyObject*,Ref> immutable_codes;
    explicit GuardPlan(PyObject* c,PyObject* h,PyObject* m):classes(c,true),helpers(h,true),missing(m,true){}
};
typedef struct {PyObject_HEAD GuardPlan* plan;} PlanObject;
PyTypeObject PlanType={PyVarObject_HEAD_INIT(nullptr,0)};

struct Pair {
    PyObject* member;PyObject* expected;
    bool operator==(const Pair& other)const{return member==other.member&&expected==other.expected;}
};
struct PairHash {
    std::size_t operator()(const Pair& value)const {
        return (reinterpret_cast<std::uintptr_t>(value.member)>>3)^reinterpret_cast<std::uintptr_t>(value.expected);
    }
};
struct MemoEntry {Ref member,expected;Result result;};
struct Context {
    GuardPlan& plan;
    std::unordered_map<Pair,MemoEntry,PairHash> memo;
    std::unordered_map<PyObject*,Ref> freevars;
    unsigned depth=0;
    explicit Context(GuardPlan& p):plan(p){}
    PyObject* names(PyObject* code) {
        auto found=freevars.find(code);if(found!=freevars.end())return found->second.p;
        Ref value(PyObject_GetAttrString(code,"co_freevars"));
        if(!value)return nullptr;
        if(!tuple(value.p)){return nullptr;}
        for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(value.p);++i)if(!text(PyTuple_GET_ITEM(value.p,i)))return nullptr;
        auto inserted=freevars.emplace(code,std::move(value));return inserted.first->second.p;
    }
};
struct Depth {
    Context& context;bool okay;
    explicit Depth(Context& c):context(c),okay(++c.depth<=256){}
    ~Depth(){--context.depth;}
};

// Preflight has no Python equality, iterator, arbitrary getattr or truth call.
// Seen sets here deduplicate grammar inspection only, never authority facts.
struct Grammar {
    std::unordered_set<PyObject*> functions,codes,dicts;
    const std::unordered_map<PyObject*,Ref>* admitted_codes=nullptr;
    std::unordered_map<PyObject*,Ref>* cold_codes=nullptr;
    explicit Grammar(const std::unordered_map<PyObject*,Ref>* admitted=nullptr,
                     std::unordered_map<PyObject*,Ref>* cold=nullptr)
        :admitted_codes(admitted),cold_codes(cold){}
    unsigned depth=0;
    bool strings_in_dict(PyObject* mapping) {
        if(!PyDict_CheckExact(mapping))return false;
        if(!dicts.insert(mapping).second)return true;
        Py_ssize_t p=0;PyObject *key,*value;
        while(PyDict_Next(mapping,&p,&key,&value))if(!text(key))return false;
        return true;
    }
    bool constant(PyObject* value) {
        if(value==Py_None||value==Py_Ellipsis||PyBool_Check(value)||PyLong_CheckExact(value)
           ||PyFloat_CheckExact(value)||PyComplex_CheckExact(value)||text(value)||PyBytes_CheckExact(value))return true;
        if(Py_IS_TYPE(value,&PyCode_Type))return code(value);
        if(PyFrozenSet_CheckExact(value)) {
            if(++depth>256){--depth;return false;}
            Ref iterator(PyObject_GetIter(value));if(!iterator){--depth;return false;}
            bool valid=true;PyObject* next;
            while(valid&&(next=PyIter_Next(iterator.p))) {Ref item(next);valid=constant(item.p);}
            if(PyErr_Occurred())valid=false;
            --depth;return valid;
        }
        if(!tuple(value))return false;
        if(++depth>256){--depth;return false;}
        bool valid=true;
        for(Py_ssize_t i=0;valid&&i<PyTuple_GET_SIZE(value);++i)valid=constant(PyTuple_GET_ITEM(value,i));
        --depth;return valid;
    }
    bool code(PyObject* value) {
        if(!Py_IS_TYPE(value,&PyCode_Type))return false;
        if(admitted_codes&&admitted_codes->find(value)!=admitted_codes->end())return true;
        if(!codes.insert(value).second)return true;
        // Decline bytes subclasses even where this CPython microversion does
        // not compare a table: no user equality may enter the C fast path.
        for(const char* attr:{"co_linetable","co_exceptiontable"}) {
            Ref table(PyObject_GetAttrString(value,attr));
            if(!table||!PyBytes_CheckExact(table.p))return false;
        }
        // CPython code equality compares constants recursively. Only immutable
        // stock constants are admitted so it cannot invoke user richcompare.
        Ref constants(PyObject_GetAttrString(value,"co_consts"));
        if(!constants||!tuple(constants.p))return false;
        for(const char* attr:{"co_name","co_qualname","co_filename"}) {
            Ref name(PyObject_GetAttrString(value,attr));if(!name||!text(name.p))return false;
        }
        for(const char* attr:{"co_names","co_freevars","co_cellvars"}) {
            Ref names(PyObject_GetAttrString(value,attr));if(!names||!tuple(names.p))return false;
            for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(names.p);++i)if(!text(PyTuple_GET_ITEM(names.p,i)))return false;
        }
        if(!constant(constants.p))return false;
        if(cold_codes)cold_codes->emplace(value,Ref(value,true));
        return true;
    }
    bool function(PyObject* value) {
        if(!exact_function(value))return true; // Opaque values are identity only.
        if(!functions.insert(value).second)return true;
        if(++depth>256){--depth;return false;}
        bool valid=code(PyFunction_GetCode(value));
        PyObject* defaults=PyFunction_GetDefaults(value);
        PyObject* keywords=PyFunction_GetKwDefaults(value);
        if(defaults&&!tuple(defaults))valid=false;
        if(keywords&&!strings_in_dict(keywords))valid=false;
        if(!strings_in_dict(PyFunction_GetGlobals(value)))valid=false;
        PyObject* closure=PyFunction_GetClosure(value);
        Ref names(PyObject_GetAttrString(PyFunction_GetCode(value),"co_freevars"));
        if(!names||!tuple(names.p)||(closure&&!tuple(closure)))valid=false;
        if(valid&&closure)for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(closure);++i) {
            PyObject* cell=PyTuple_GET_ITEM(closure,i);
            if(!PyCell_Check(cell)){valid=false;break;}
            PyObject* current=PyCell_GET(cell);
            if(current&&!function(current)){valid=false;break;}
            if(i<PyTuple_GET_SIZE(names.p)) {
                PyObject* name=PyTuple_GET_ITEM(names.p,i);
                if(!text(name)){valid=false;break;}
                if(current&&PyLong_CheckExact(current)&&PyUnicode_CompareWithASCIIString(name,"accepted_generation")==0) {
                    Ref module(PyObject_GetAttrString(value,"__module__"));
                    Ref qualname(PyObject_GetAttrString(value,"__qualname__"));
                    if(!module||!qualname||!text(module.p)||!text(qualname.p)){valid=false;break;}
                }
            }
        }
        --depth;return valid;
    }
    bool member(PyObject* value) {
        if(exact_function(value))return function(value);
        if(Py_IS_TYPE(value,&PyProperty_Type)) {
            for(const char* name:{"fget","fset","fdel"}) {
                Ref child(PyObject_GetAttrString(value,name));
                if(!child||!function(child.p))return false;
            }
        } else if(Py_IS_TYPE(value,&PyStaticMethod_Type)||Py_IS_TYPE(value,&PyClassMethod_Type)) {
            Ref child(PyObject_GetAttrString(value,"__func__"));
            if(!child||!function(child.p))return false;
        }
        return true;
    }
    bool classes(GuardPlan& plan) {
        for(auto& record:plan.records) {
            if(!type_access_supported(record.cls.p))return false;
            auto cls=reinterpret_cast<PyTypeObject*>(record.cls.p);
            if(!cls->tp_mro||!tuple(cls->tp_mro))return false;
            for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(cls->tp_mro);++i) {
                PyObject* base=PyTuple_GET_ITEM(cls->tp_mro,i);
                if(!type_access_supported(base))return false;
                PyObject* ns=reinterpret_cast<PyTypeObject*>(base)->tp_dict;
                if(!strings_in_dict(ns))return false;
                Py_ssize_t p=0;PyObject *key,*value;
                while(PyDict_Next(ns,&p,&key,&value))if(!member(value))return false;
            }
        }
        for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(plan.helpers.p);++i) {
            PyObject* row=PyTuple_GET_ITEM(plan.helpers.p,i);
            if(!tuple(row,3)||!member(PyTuple_GET_ITEM(row,0)))return false;
        }
        return true;
    }
};

bool expected_state(Grammar& grammar,PyObject* state,unsigned depth=0) {
    if(depth>256)return false;
    if(PyLong_CheckExact(state)||text(state))return true;
    if(!tuple(state))return false;
    Py_ssize_t size=PyTuple_GET_SIZE(state);
    if(size==2)return PyType_Check(PyTuple_GET_ITEM(state,0))&&PyLong_CheckExact(PyTuple_GET_ITEM(state,1));
    if(size==4) {
        if(!PyLong_CheckExact(PyTuple_GET_ITEM(state,0)))return false;
        for(Py_ssize_t i=1;i<4;++i)if(!expected_state(grammar,PyTuple_GET_ITEM(state,i),depth+1))return false;
        return true;
    }
    if(size!=7||!grammar.code(PyTuple_GET_ITEM(state,1)))return false;
    for(int i:{0,2,4})if(!PyLong_CheckExact(PyTuple_GET_ITEM(state,i)))return false;
    PyObject* defaults=PyTuple_GET_ITEM(state,3);
    PyObject* keywords=PyTuple_GET_ITEM(state,5);
    PyObject* cells=PyTuple_GET_ITEM(state,6);
    if(!tuple(defaults)||!tuple(keywords)||!tuple(cells))return false;
    for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(defaults);++i)if(!PyLong_CheckExact(PyTuple_GET_ITEM(defaults,i)))return false;
    for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(keywords);++i) {
        PyObject* pair=PyTuple_GET_ITEM(keywords,i);
        if(!tuple(pair,2)||!text(PyTuple_GET_ITEM(pair,0))||!PyLong_CheckExact(PyTuple_GET_ITEM(pair,1)))return false;
    }
    for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(cells);++i)if(!expected_state(grammar,PyTuple_GET_ITEM(cells,i),depth+1))return false;
    return true;
}
bool expected_classes(GuardPlan& plan) {
    Grammar grammar;
    for(auto& record:plan.records)for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(record.expected.p);++i) {
        PyObject* row=PyTuple_GET_ITEM(record.expected.p,i);
        if(!tuple(row,3)||!PyType_Check(PyTuple_GET_ITEM(row,0))||!text(PyTuple_GET_ITEM(row,1))
           ||!expected_state(grammar,PyTuple_GET_ITEM(row,2)))return false;
    }
    return true;
}

// Literal predicates resolve globals before their original function-builtins
// dictionaries. Capture both presence and identity, including absent shadows.
bool canonical_builtin(PyObject* name,PyObject* value) {
    if(!text(name))return false;
    const char* s=PyUnicode_AsUTF8(name);if(!s)return false;
    struct TypeName {const char* name;PyObject* object;};
    const TypeName known[]={
        {"type",reinterpret_cast<PyObject*>(&PyType_Type)}, {"int",reinterpret_cast<PyObject*>(&PyLong_Type)},
        {"str",reinterpret_cast<PyObject*>(&PyUnicode_Type)}, {"dict",reinterpret_cast<PyObject*>(&PyDict_Type)},
        {"tuple",reinterpret_cast<PyObject*>(&PyTuple_Type)}, {"set",reinterpret_cast<PyObject*>(&PySet_Type)},
        {"property",reinterpret_cast<PyObject*>(&PyProperty_Type)}, {"staticmethod",reinterpret_cast<PyObject*>(&PyStaticMethod_Type)},
        {"classmethod",reinterpret_cast<PyObject*>(&PyClassMethod_Type)}, {"ValueError",PyExc_ValueError}};
    for(auto item:known)if(std::strcmp(s,item.name)==0)return value==item.object;
    for(const char* function:{"id","vars","len","all","any","zip"})if(std::strcmp(s,function)==0) {
        if(std::strcmp(s,"zip")==0)return value==reinterpret_cast<PyObject*>(&PyZip_Type);
        if(!Py_IS_TYPE(value,&PyCFunction_Type))return false;
        auto c=reinterpret_cast<PyCFunctionObject*>(value);
        if(std::strcmp(c->m_ml->ml_name,function)!=0)return false;
        PyObject* self=PyCFunction_GetSelf(value);
        if(!self||!PyModule_CheckExact(self))return false;
        PyObject* owner_name=nullptr;
        if(!namespace_lookup(PyModule_GetDict(self),"__name__",owner_name)||!owner_name||!text(owner_name))return false;
        return PyUnicode_CompareWithASCIIString(owner_name,"builtins")==0;
    }
    return true; // Other names are still bound by exact presence/value identity.
}
bool add_binding(EnvironmentRecord& record,PyObject* mapping,PyObject* name) {
    PyObject* value=PyDict_GetItemWithError(mapping,name);
    if(!value&&PyErr_Occurred())return false;
    Ref binding(PyTuple_Pack(4,mapping,name,value?Py_True:Py_False,value?value:Py_None));
    if(!binding)return false;record.bindings.push_back(std::move(binding));return true;
}
bool capture_environment(GuardPlan& plan,PyObject* roots) {
    if(!tuple(roots))return false;
    Grammar grammar;
    for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(roots);++i) {
        PyObject* function=PyTuple_GET_ITEM(roots,i);
        if(!exact_function(function)||!grammar.function(function))return false;
        EnvironmentRecord record;
        record.function=Ref(function,true);record.code=Ref(PyFunction_GetCode(function),true);
        record.globals=Ref(PyFunction_GetGlobals(function),true);
        PyObject* defaults=PyFunction_GetDefaults(function);record.defaults=Ref(defaults?defaults:Py_None,true);
        if(defaults)for(Py_ssize_t j=0;j<PyTuple_GET_SIZE(defaults);++j)record.default_values.push_back(Ref(PyTuple_GET_ITEM(defaults,j),true));
        PyObject* keywords=PyFunction_GetKwDefaults(function);record.keywords=Ref(keywords?keywords:Py_None,true);
        if(keywords) {
            Py_ssize_t p=0;PyObject *key,*value;
            while(PyDict_Next(keywords,&p,&key,&value)) {
                Ref pair(PyTuple_Pack(2,key,value));if(!pair)return false;record.keyword_pairs.push_back(std::move(pair));
            }
        }
        PyObject* closure=PyFunction_GetClosure(function);record.closure=Ref(closure?closure:Py_None,true);
        if(closure)for(Py_ssize_t j=0;j<PyTuple_GET_SIZE(closure);++j)record.cell_values.push_back(Ref(PyCell_GET(PyTuple_GET_ITEM(closure,j)),true));
        record.builtins=Ref(PyObject_GetAttrString(function,"__builtins__"));
        if(!record.builtins||!grammar.strings_in_dict(record.builtins.p))return false;
        std::vector<Ref> work;work.push_back(record.code);
        std::unordered_set<PyObject*> visited;
        while(!work.empty()) {
            Ref code=std::move(work.back());work.pop_back();if(!visited.insert(code.p).second)continue;
            Ref names(PyObject_GetAttrString(code.p,"co_names"));
            Ref constants(PyObject_GetAttrString(code.p,"co_consts"));
            if(!names||!constants||!tuple(names.p)||!tuple(constants.p))return false;
            for(Py_ssize_t j=0;j<PyTuple_GET_SIZE(names.p);++j) {
                PyObject* name=PyTuple_GET_ITEM(names.p,j);if(!text(name))return false;
                if(!add_binding(record,record.globals.p,name))return false;
                PyObject* value=PyDict_GetItemWithError(record.globals.p,name);
                if(!value&&PyErr_Occurred())return false;
                if(!value) {
                    value=PyDict_GetItemWithError(record.builtins.p,name);
                    if(!value&&PyErr_Occurred())return false;
                    if(value&&!canonical_builtin(name,value))return false;
                    if(!add_binding(record,record.builtins.p,name))return false;
                } else if(!canonical_builtin(name,value))return false;
                if(value&&PyUnicode_CompareWithASCIIString(name,"types")==0) {
                    if(!PyModule_CheckExact(value))return false;
                    PyObject* mapping=PyModule_GetDict(value);
                    if(!grammar.strings_in_dict(mapping))return false;
                    Ref key(PyUnicode_FromString("FunctionType"));if(!key)return false;
                    if(PyDict_GetItemWithError(mapping,key.p)!=reinterpret_cast<PyObject*>(&PyFunction_Type))return false;
                    if(!add_binding(record,mapping,key.p))return false;
                }
            }
            for(Py_ssize_t j=0;j<PyTuple_GET_SIZE(constants.p);++j) {
                PyObject* item=PyTuple_GET_ITEM(constants.p,j);
                if(Py_IS_TYPE(item,&PyCode_Type))work.push_back(Ref(item,true));
            }
        }
        plan.environments.push_back(std::move(record));
    }
    return true;
}
Result environment(GuardPlan& plan) {
    Grammar grammar(&plan.immutable_codes);
    for(auto& record:plan.environments) {
        if(!exact_function(record.function.p)||PyFunction_GetCode(record.function.p)!=record.code.p
           ||PyFunction_GetGlobals(record.function.p)!=record.globals.p)return Result::Unsupported;
        Ref builtins(PyObject_GetAttrString(record.function.p,"__builtins__"));
        if(!builtins)return Result::Error;if(builtins.p!=record.builtins.p)return Result::Unsupported;
        PyObject* defaults=PyFunction_GetDefaults(record.function.p);if(!defaults)defaults=Py_None;
        PyObject* keywords=PyFunction_GetKwDefaults(record.function.p);if(!keywords)keywords=Py_None;
        PyObject* closure=PyFunction_GetClosure(record.function.p);if(!closure)closure=Py_None;
        if(defaults!=record.defaults.p||keywords!=record.keywords.p||closure!=record.closure.p)return Result::Unsupported;
        if(defaults!=Py_None) {
            if(!tuple(defaults)||PyTuple_GET_SIZE(defaults)!=static_cast<Py_ssize_t>(record.default_values.size()))return Result::Unsupported;
            for(Py_ssize_t j=0;j<PyTuple_GET_SIZE(defaults);++j)if(PyTuple_GET_ITEM(defaults,j)!=record.default_values[j].p)return Result::Unsupported;
        }
        if(keywords!=Py_None) {
            if(!grammar.strings_in_dict(keywords))return Result::Unsupported;
            Py_ssize_t p=0,j=0;PyObject *key,*value;
            while(PyDict_Next(keywords,&p,&key,&value)) {
                if(j>=static_cast<Py_ssize_t>(record.keyword_pairs.size()))return Result::Unsupported;
                PyObject* pair=record.keyword_pairs[j++].p;
                if(key!=PyTuple_GET_ITEM(pair,0)||value!=PyTuple_GET_ITEM(pair,1))return Result::Unsupported;
            }
            if(j!=static_cast<Py_ssize_t>(record.keyword_pairs.size()))return Result::Unsupported;
        }
        if(closure!=Py_None) {
            if(!tuple(closure)||PyTuple_GET_SIZE(closure)!=static_cast<Py_ssize_t>(record.cell_values.size()))return Result::Unsupported;
            for(Py_ssize_t j=0;j<PyTuple_GET_SIZE(closure);++j) {
                PyObject* cell=PyTuple_GET_ITEM(closure,j);
                if(!PyCell_Check(cell)||PyCell_GET(cell)!=record.cell_values[j].p)return Result::Unsupported;
            }
        }
        for(auto& ref:record.bindings) {
            PyObject* binding=ref.p;PyObject* mapping=PyTuple_GET_ITEM(binding,0);
            if(!grammar.strings_in_dict(mapping))return Result::Unsupported;
            PyObject* value=PyDict_GetItemWithError(mapping,PyTuple_GET_ITEM(binding,1));
            if(!value&&PyErr_Occurred())return Result::Error;
            bool present=PyTuple_GET_ITEM(binding,2)==Py_True;
            if((value!=nullptr)!=present||(present&&value!=PyTuple_GET_ITEM(binding,3)))return Result::Unsupported;
        }
    }
    return Result::Equal;
}

Result keyword_state(PyObject* current,PyObject* expected) {
    // _callable_state emits ordered (key,id(value)) tuples.
    if(!tuple(expected))return Result::Unsupported;
    if(!current||current==Py_None)return PyTuple_GET_SIZE(expected)==0?Result::Equal:Result::Changed;
    if(!PyDict_CheckExact(current))return Result::Unsupported;
    Py_ssize_t position=0,index=0;PyObject *key,*value;
    while(PyDict_Next(current,&position,&key,&value)) {
        if(!text(key))return Result::Unsupported;
        if(index>=PyTuple_GET_SIZE(expected))return Result::Changed;
        PyObject* pair=PyTuple_GET_ITEM(expected,index++);
        if(!tuple(pair,2)||!text(PyTuple_GET_ITEM(pair,0)))return Result::Unsupported;
        int equal=PyObject_RichCompareBool(key,PyTuple_GET_ITEM(pair,0),Py_EQ);
        if(equal<0)return Result::Error;if(!equal)return Result::Changed;
        Result result=object_id(PyTuple_GET_ITEM(pair,1),value);if(result!=Result::Equal)return result;
    }
    return index==PyTuple_GET_SIZE(expected)?Result::Equal:Result::Changed;
}

Result owned_counter(PyObject* fn,PyObject* name,PyObject* value,bool& owned) {
    owned=false;
    Result named=string_equal(name,"accepted_generation");
    if(named==Result::Error||named==Result::Unsupported)return named;
    if(named!=Result::Equal||!PyLong_CheckExact(value))return Result::Equal;
    Ref module(PyObject_GetAttrString(fn,"__module__"));
    Ref qualname(PyObject_GetAttrString(fn,"__qualname__"));
    if(!module||!qualname)return Result::Error;
    Result mod=string_equal(module.p,"anyfesolver._qualified_authority_epoch");
    if(mod==Result::Error||mod==Result::Unsupported)return mod;
    if(mod!=Result::Equal)return Result::Equal;
    if(!text(qualname.p))return Result::Unsupported;
    for(const char* target:{"_AuthorityEpochManager.bind.<locals>.require",
            "_AuthorityEpochManager.bind_context.<locals>.require","_AuthorityEpochManager.bind_argument.<locals>.require"}) {
        Result same=string_equal(qualname.p,target);if(same==Result::Error)return same;
        if(same==Result::Equal){owned=true;break;}
    }
    return Result::Equal;
}

Result callable_state(Context& context,PyObject* current,PyObject* expected,
                      std::unordered_set<PyObject*>& seen) {
    ++statistics.callable_nodes;
    Depth depth(context);if(!depth.okay)return Result::Unsupported;
    if(!exact_function(current)) {
        if(!tuple(expected,2)||!PyType_Check(PyTuple_GET_ITEM(expected,0)))return Result::Unsupported;
        PyObject* kind=reinterpret_cast<PyObject*>(Py_TYPE(current));
        PyObject* expected_kind=PyTuple_GET_ITEM(expected,0);
        if(kind!=expected_kind) {
            if(Py_TYPE(kind)->tp_richcompare!=PyType_Type.tp_richcompare||Py_TYPE(expected_kind)->tp_richcompare!=PyType_Type.tp_richcompare)return Result::Unsupported;
            int same=PyObject_RichCompareBool(kind,expected_kind,Py_EQ);if(same<0)return Result::Error;
            if(!same)return Result::Changed;
        }
        return object_id(PyTuple_GET_ITEM(expected,1),current);
    }
    if(seen.find(current)!=seen.end())return object_id(expected,current);
    seen.insert(current);
    if(!tuple(expected,7))return Result::Unsupported;
    Result result=object_id(PyTuple_GET_ITEM(expected,0),current);if(result!=Result::Equal)return result;
    PyObject* code=PyFunction_GetCode(current);
    PyObject* captured_code=PyTuple_GET_ITEM(expected,1);
    if(!PyCode_Check(code)||!PyCode_Check(captured_code))return Result::Unsupported;
    int same=PyObject_RichCompareBool(code,captured_code,Py_EQ);
    if(same<0)return Result::Error;if(!same)return Result::Changed;
    PyObject* defaults=PyFunction_GetDefaults(current);if(!defaults)defaults=Py_None;
    result=object_id(PyTuple_GET_ITEM(expected,2),defaults);if(result!=Result::Equal)return result;
    PyObject* items=PyTuple_GET_ITEM(expected,3);
    if(!tuple(items)||(defaults!=Py_None&&!tuple(defaults)))return Result::Unsupported;
    Py_ssize_t count=defaults==Py_None?0:PyTuple_GET_SIZE(defaults);
    if(count!=PyTuple_GET_SIZE(items))return Result::Changed;
    for(Py_ssize_t i=0;i<count;++i){result=object_id(PyTuple_GET_ITEM(items,i),PyTuple_GET_ITEM(defaults,i));if(result!=Result::Equal)return result;}
    PyObject* keywords=PyFunction_GetKwDefaults(current);if(!keywords)keywords=Py_None;
    result=object_id(PyTuple_GET_ITEM(expected,4),keywords);if(result!=Result::Equal)return result;
    result=keyword_state(keywords,PyTuple_GET_ITEM(expected,5));if(result!=Result::Equal)return result;
    PyObject* closure=PyFunction_GetClosure(current);
    PyObject* names=context.names(code);if(!names)return PyErr_Occurred()?Result::Error:Result::Unsupported;
    PyObject* cells=PyTuple_GET_ITEM(expected,6);
    if(!tuple(cells)||(closure&&!tuple(closure)))return Result::Unsupported;
    count=closure?PyTuple_GET_SIZE(closure):0;
    Py_ssize_t zipped=std::min(count,PyTuple_GET_SIZE(names));
    if(zipped!=PyTuple_GET_SIZE(cells))return Result::Changed;
    for(Py_ssize_t i=0;i<zipped;++i) {
        PyObject* cell=PyTuple_GET_ITEM(closure,i);if(!PyCell_Check(cell))return Result::Unsupported;
        PyObject* value=PyCell_GET(cell);PyObject* captured=PyTuple_GET_ITEM(cells,i);
        if(!value){result=string_equal(captured,"empty");if(result!=Result::Equal)return result;continue;}
        bool owned=false;result=owned_counter(current,PyTuple_GET_ITEM(names,i),value,owned);if(result!=Result::Equal)return result;
        if(owned)result=string_equal(captured,"owned_epoch_counter");
        else if(exact_function(value))result=callable_state(context,value,captured,seen);
        else result=object_id(captured,value);
        if(result!=Result::Equal)return result;
    }
    return Result::Equal;
}

Result root_state(Context& context,PyObject* member,PyObject* expected) {
    std::unordered_set<PyObject*> seen;return callable_state(context,member,expected,seen);
}
Result class_state(Context& context,PyObject* cls,PyObject* expected) {
    if(!type_access_supported(cls)||!tuple(expected))return Result::Unsupported;
    PyTypeObject* current=reinterpret_cast<PyTypeObject*>(cls);
    if(!current->tp_mro||!tuple(current->tp_mro))return Result::Unsupported;
    Py_ssize_t output=0;
    for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(current->tp_mro);++i) {
        PyObject* base=PyTuple_GET_ITEM(current->tp_mro,i);
        if(!type_access_supported(base))return Result::Unsupported;
        PyObject* ns=reinterpret_cast<PyTypeObject*>(base)->tp_dict;
        if(!ns||!PyDict_CheckExact(ns))return Result::Unsupported;
        // _class_state records each MRO namespace in member-name order. Snapshot
        // the pairs before sorting. Exact, distinct Unicode keys decide every
        // comparison between entries, so no member comparison hook can run.
        Ref members(PyDict_Items(ns));
        if(!members)return Result::Error;
        const Py_ssize_t count=PyList_GET_SIZE(members.p);
        for(Py_ssize_t j=0;j<count;++j) {
            if(!text(PyTuple_GET_ITEM(PyList_GET_ITEM(members.p,j),0)))return Result::Unsupported;
        }
        if(PyList_Sort(members.p)<0)return Result::Error;
        for(Py_ssize_t j=0;j<count;++j) {
            PyObject* pair=PyList_GET_ITEM(members.p,j);
            PyObject* name=PyTuple_GET_ITEM(pair,0);
            PyObject* value=PyTuple_GET_ITEM(pair,1);
            Result slots=string_equal(name,"__slotnames__");if(slots==Result::Error)return slots;
            if(slots==Result::Equal)continue;
            if(output>=PyTuple_GET_SIZE(expected))return Result::Changed;
            PyObject* row=PyTuple_GET_ITEM(expected,output++);
            if(!tuple(row,3)||!text(PyTuple_GET_ITEM(row,1)))return Result::Unsupported;
            // Same-base equality is identity in this supported grammar.
            if(PyTuple_GET_ITEM(row,0)!=base)return Result::Changed;
            int named=PyObject_RichCompareBool(name,PyTuple_GET_ITEM(row,1),Py_EQ);
            if(named<0)return Result::Error;if(!named)return Result::Changed;
            PyObject* state=PyTuple_GET_ITEM(row,2);Result result;
            if(exact_function(value))result=root_state(context,value,state);
            else if(Py_IS_TYPE(value,&PyProperty_Type)) {
                if(!tuple(state,4))return Result::Unsupported;
                result=object_id(PyTuple_GET_ITEM(state,0),value);if(result!=Result::Equal)return result;
                const char* attrs[]={"fget","fset","fdel"};
                for(int part=0;part<3;++part) {
                    Ref child(PyObject_GetAttrString(value,attrs[part]));if(!child)return Result::Error;
                    result=root_state(context,child.p,PyTuple_GET_ITEM(state,part+1));if(result!=Result::Equal)return result;
                }
            } else if(Py_IS_TYPE(value,&PyStaticMethod_Type)||Py_IS_TYPE(value,&PyClassMethod_Type)) {
                Ref child(PyObject_GetAttrString(value,"__func__"));if(!child)return Result::Error;
                result=root_state(context,child.p,state);
            } else {
                if(!tuple(state,2))return Result::Unsupported;
                PyObject* kind=reinterpret_cast<PyObject*>(Py_TYPE(value));
                PyObject* captured=PyTuple_GET_ITEM(state,0);
                if(!PyType_Check(captured))return Result::Unsupported;
                if(kind!=captured) {
                    if(Py_TYPE(kind)->tp_richcompare!=PyType_Type.tp_richcompare||Py_TYPE(captured)->tp_richcompare!=PyType_Type.tp_richcompare)return Result::Unsupported;
                    int matched=PyObject_RichCompareBool(kind,captured,Py_EQ);if(matched<0)return Result::Error;
                    if(!matched)return Result::Changed;
                }
                result=object_id(PyTuple_GET_ITEM(state,1),value);
            }
            if(result!=Result::Equal)return result;
        }
    }
    return output==PyTuple_GET_SIZE(expected)?Result::Equal:Result::Changed;
}

Result authority(Context& context,PyObject* member,PyObject* expected) {
    ++statistics.authority_nodes;
    Depth depth(context);if(!depth.okay)return Result::Unsupported;
    if(!tuple(expected))return Result::Unsupported;
    Pair key{member,expected};auto cached=context.memo.find(key);
    if(cached!=context.memo.end())return cached->second.result;
    Result result=Result::Equal;
    if(exact_function(member)) {
        Py_ssize_t length=PyTuple_GET_SIZE(expected);
        if(length==1)result=member==PyTuple_GET_ITEM(expected,0)?Result::Equal:Result::Changed;
        else if(length!=5)result=Result::Changed;
        else {
            PyObject* defaults=PyFunction_GetDefaults(member);if(!defaults)defaults=Py_None;
            if(member!=PyTuple_GET_ITEM(expected,0)||PyFunction_GetCode(member)!=PyTuple_GET_ITEM(expected,1)||defaults!=PyTuple_GET_ITEM(expected,2))result=Result::Changed;
            else {
                PyObject* closure=PyFunction_GetClosure(member);PyObject* cells=PyTuple_GET_ITEM(expected,4);
                if((closure&&!tuple(closure))||!tuple(cells))return Result::Unsupported;
                Py_ssize_t size=closure?PyTuple_GET_SIZE(closure):0;
                if(size!=PyTuple_GET_SIZE(cells))result=Result::Changed;
                for(Py_ssize_t i=0;result==Result::Equal&&i<size;++i) {
                    PyObject* cell=PyTuple_GET_ITEM(closure,i);PyObject* captured=PyTuple_GET_ITEM(cells,i);
                    if(!PyCell_Check(cell)||!tuple(captured,3)||!PyBool_Check(PyTuple_GET_ITEM(captured,2)))return Result::Unsupported;
                    PyObject* value=PyCell_GET(cell);
                    if(!value){result=Result::Changed;break;}
                    if(PyTuple_GET_ITEM(captured,2)==Py_True)result=PyLong_CheckExact(value)?Result::Equal:Result::Changed;
                    else if(value!=PyTuple_GET_ITEM(captured,0))result=Result::Changed;
                    else result=authority(context,value,PyTuple_GET_ITEM(captured,1));
                }
                if(result==Result::Equal) {
                    PyObject* keywords=PyFunction_GetKwDefaults(member);if(!keywords)keywords=Py_None;
                    PyObject* captured=PyTuple_GET_ITEM(expected,3);
                    if(captured==Py_None)result=keywords==Py_None?Result::Equal:Result::Changed;
                    else if(!tuple(captured))return Result::Unsupported;
                    else if(!PyDict_CheckExact(keywords))result=Result::Changed;
                    else {
                        Py_ssize_t p=0;PyObject *k,*v;
                        while(PyDict_Next(keywords,&p,&k,&v))if(!text(k))return Result::Unsupported;
                        if(PyDict_Size(keywords)!=PyTuple_GET_SIZE(captured))result=Result::Changed;
                        for(Py_ssize_t i=0;result==Result::Equal&&i<PyTuple_GET_SIZE(captured);++i) {
                            PyObject* item=PyTuple_GET_ITEM(captured,i);
                            if(!tuple(item,2)||!text(PyTuple_GET_ITEM(item,0)))return Result::Unsupported;
                            PyObject* actual=PyDict_GetItemWithError(keywords,PyTuple_GET_ITEM(item,0));
                            if(!actual&&PyErr_Occurred())return Result::Error;
                            if(!actual)actual=context.plan.missing.p;
                            if(actual!=PyTuple_GET_ITEM(item,1))result=Result::Changed;
                        }
                    }
                }
            }
        }
    } else if(Py_IS_TYPE(member,&PyProperty_Type)) {
        if(!tuple(expected,3))result=Result::Changed;
        else {const char* attrs[]={"fget","fset","fdel"};for(int i=0;i<3&&result==Result::Equal;++i){Ref value(PyObject_GetAttrString(member,attrs[i]));if(!value)return Result::Error;result=authority(context,value.p,PyTuple_GET_ITEM(expected,i));}}
    } else if(Py_IS_TYPE(member,&PyStaticMethod_Type)||Py_IS_TYPE(member,&PyClassMethod_Type)) {
        if(!tuple(expected,1))result=Result::Changed;
        else {Ref value(PyObject_GetAttrString(member,"__func__"));if(!value)return Result::Error;result=authority(context,value.p,PyTuple_GET_ITEM(expected,0));}
    } else result=PyTuple_GET_SIZE(expected)==0?Result::Equal:Result::Changed;
    context.memo.emplace(key,MemoEntry{Ref(member,true),Ref(expected,true),result});return result;
}

Result globals(Context& context,PyObject* member,PyObject* bindings) {
    if(!exact_function(member)||!tuple(bindings))return Result::Unsupported;
    PyObject* ns=PyFunction_GetGlobals(member);if(!PyDict_CheckExact(ns))return Result::Unsupported;
    for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(bindings);++i) {
        ++statistics.global_bindings;
        PyObject* item=PyTuple_GET_ITEM(bindings,i);
        if(!tuple(item)||!text(PyTuple_GET_ITEM(item,0)))return Result::Unsupported;
        PyObject* value=PyDict_GetItemWithError(ns,PyTuple_GET_ITEM(item,0));
        if(!value&&PyErr_Occurred())return Result::Error;
        if(tuple(item,2)) {
            if(!value)value=Py_None; // Literal .get(name) missing-None policy.
            if(value!=PyTuple_GET_ITEM(item,1))return Result::Changed;
        } else if(tuple(item,3)&&PyBool_Check(PyTuple_GET_ITEM(item,1))) {
            bool present=PyTuple_GET_ITEM(item,1)==Py_True;
            if((value!=nullptr)!=present||(present&&value!=PyTuple_GET_ITEM(item,2)))return Result::Changed;
        } else return Result::Unsupported;
    }
    return Result::Equal;
}

// Grammar/environment preflight is intentionally separate from comparison.
// Integration supplies validated literal-environment refs. Unknown routing
// returns NotImplemented; caller must run the complete literal guard once.
Result owner(GuardPlan& plan,PyObject* classes,PyObject* helpers) {
    if(classes!=plan.classes.p||helpers!=plan.helpers.p||!PyDict_CheckExact(classes))return Result::Changed;
    Py_ssize_t p=0,index=0;PyObject *key,*value;
    while(PyDict_Next(classes,&p,&key,&value)) {
        if(index>=static_cast<Py_ssize_t>(plan.records.size()))return Result::Changed;
        auto& record=plan.records[index++];if(key!=record.cls.p||value!=record.expected.p)return Result::Changed;
    }
    return index==static_cast<Py_ssize_t>(plan.records.size())?Result::Equal:Result::Changed;
}

PyObject* make_plan(PyObject*,PyObject* args) {
    PyObject *classes,*helpers,*missing,*literal_roots;
    if(!PyArg_ParseTuple(args,"OOOO",&classes,&helpers,&missing,&literal_roots))return nullptr;
    if(!PyDict_CheckExact(classes)||!tuple(helpers)){Py_INCREF(Py_NotImplemented);return Py_NotImplemented;}
    try {
        auto data=std::make_unique<GuardPlan>(classes,helpers,missing);
        Py_ssize_t p=0;PyObject *key,*value;
        while(PyDict_Next(classes,&p,&key,&value)) {
            if(!PyType_Check(key)||!tuple(value)){Py_INCREF(Py_NotImplemented);return Py_NotImplemented;}
            data->records.push_back(ClassRecord{Ref(key,true),Ref(value,true)});
        }
        if(!expected_classes(*data)||!capture_environment(*data,literal_roots)) {
            if(PyErr_Occurred())return nullptr;
            Py_INCREF(Py_NotImplemented);return Py_NotImplemented;
        }
        // Unsupported current grammar keeps the original admitted plan/fallback.
        // Only independently completed exact-CodeType classifications survive.
        Grammar cold(nullptr,&data->immutable_codes);
        cold.classes(*data);
        if(PyErr_Occurred())return nullptr;
        auto object=PyObject_GC_New(PlanObject,&PlanType);
        if(!object)return nullptr;object->plan=data.release();PyObject_GC_Track(object);
        return reinterpret_cast<PyObject*>(object);
    } catch(const std::bad_alloc&){return PyErr_NoMemory();}
}
PyObject* compare(PyObject*,PyObject* args) {
    PyObject *object,*classes,*helpers;
    if(!PyArg_ParseTuple(args,"OOO",&object,&classes,&helpers))return nullptr;
    if(!Py_IS_TYPE(object,&PlanType)){PyErr_SetString(PyExc_TypeError,"native guard plan type mismatch");return nullptr;}
    auto data=reinterpret_cast<PlanObject*>(object)->plan;
    if(!data){PyErr_SetString(PyExc_RuntimeError,"native guard plan has been cleared");return nullptr;}
    auto& plan=*data;
    ++statistics.calls;
    try {
        if(owner(plan,classes,helpers)!=Result::Equal){++statistics.owner_changes;return PyLong_FromLong(2);} // Admitted owner tamper.
        Result env=environment(plan);
        if(env==Result::Error){++statistics.errors;return nullptr;}
        Grammar grammar(&plan.immutable_codes);
        if(env!=Result::Equal||!grammar.classes(plan)) {
            if(PyErr_Occurred()){++statistics.errors;return nullptr;}
            ++statistics.unsupported;
            Py_INCREF(Py_NotImplemented);return Py_NotImplemented;
        }
        Context context(plan);
        Result result=Result::Equal;
        for(auto& record:plan.records){++statistics.classes;result=class_state(context,record.cls.p,record.expected.p);if(result!=Result::Equal)break;}
        for(Py_ssize_t i=0;result==Result::Equal&&i<PyTuple_GET_SIZE(helpers);++i) {
            ++statistics.helpers;
            PyObject* row=PyTuple_GET_ITEM(helpers,i);if(!tuple(row,3)){result=Result::Unsupported;break;}
            result=authority(context,PyTuple_GET_ITEM(row,0),PyTuple_GET_ITEM(row,1));
            if(result==Result::Equal)result=globals(context,PyTuple_GET_ITEM(row,0),PyTuple_GET_ITEM(row,2));
        }
        if(result==Result::Error){++statistics.errors;return nullptr;}
        if(result==Result::Unsupported){++statistics.unsupported;Py_INCREF(Py_NotImplemented);return Py_NotImplemented;}
        if(result==Result::Equal)++statistics.passes;else ++statistics.mismatches;
        return PyBool_FromLong(result==Result::Equal);
    } catch(const std::bad_alloc&){++statistics.errors;return PyErr_NoMemory();}
}
int plan_traverse(PlanObject* object,visitproc visit,void* arg) {
    auto plan=object->plan;if(!plan)return 0;
    Py_VISIT(plan->classes.p);Py_VISIT(plan->helpers.p);Py_VISIT(plan->missing.p);
    for(auto& item:plan->records){Py_VISIT(item.cls.p);Py_VISIT(item.expected.p);}
    for(auto& item:plan->immutable_codes)Py_VISIT(item.second.p);
    for(auto& item:plan->environments) {
        Py_VISIT(item.function.p);Py_VISIT(item.code.p);Py_VISIT(item.globals.p);Py_VISIT(item.builtins.p);
        Py_VISIT(item.defaults.p);Py_VISIT(item.keywords.p);Py_VISIT(item.closure.p);
        for(auto& value:item.default_values)Py_VISIT(value.p);
        for(auto& pair:item.keyword_pairs)Py_VISIT(pair.p);
        for(auto& value:item.cell_values)Py_VISIT(value.p);
        for(auto& binding:item.bindings)Py_VISIT(binding.p);
    }
    return 0;
}
int plan_clear(PlanObject* object){auto plan=object->plan;object->plan=nullptr;delete plan;return 0;}
void plan_dealloc(PlanObject* object){PyObject_GC_UnTrack(object);plan_clear(object);PyObject_GC_Del(object);}
PyObject* build_info(PyObject*,PyObject*) {
    return Py_BuildValue("(iss)",1,"cp311-win_amd64",ANYSOLVER_GUARD_SOURCE_SHA);
}
PyObject* binary_identity(PyObject*,PyObject*) {
#ifdef _WIN32
    HMODULE handle=nullptr;
    if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&compare),&handle)){PyErr_SetFromWindowsErr(0);return nullptr;}
    wchar_t path[32768];DWORD size=GetModuleFileNameW(handle,path,32768);
    if(!size||size>=32768){PyErr_SetFromWindowsErr(0);return nullptr;}
    Ref name(PyUnicode_FromWideChar(path,size));if(!name)return nullptr;
    return Py_BuildValue("(OKK)",name.p,static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(handle)),
        static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(&compare)));
#else
    Py_INCREF(Py_NotImplemented);return Py_NotImplemented;
#endif
}
extern PyMethodDef methods[];
PyObject* entry_identity(PyObject*,PyObject* entry) {
    if(!Py_IS_TYPE(entry,&PyCFunction_Type)){Py_INCREF(Py_NotImplemented);return Py_NotImplemented;}
    auto function=reinterpret_cast<PyCFunctionObject*>(entry);
    PyMethodDef* definition=function->m_ml;
    const char* name=nullptr;
    for(PyMethodDef* candidate=methods;candidate->ml_name;++candidate)if(definition==candidate){name=candidate->ml_name;break;}
    PyObject* owner=PyCFunction_GetSelf(entry);
    if(!name||!owner||!PyModule_CheckExact(owner)){Py_INCREF(Py_NotImplemented);return Py_NotImplemented;}
    return Py_BuildValue("(sOKKi)",name,owner,
        static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(definition)),
        static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(definition->ml_meth)),definition->ml_flags);
}
PyObject* get_statistics(PyObject*,PyObject*) {
    return Py_BuildValue("{sK,sK,sK,sK,sK,sK,sK,sK,sK,sK,sK}",
        "calls",statistics.calls,"passes",statistics.passes,"mismatches",statistics.mismatches,
        "unsupported",statistics.unsupported,"owner_changes",statistics.owner_changes,"errors",statistics.errors,
        "class_facts",statistics.classes,"helper_facts",statistics.helpers,"callable_nodes",statistics.callable_nodes,
        "authority_nodes",statistics.authority_nodes,"global_bindings",statistics.global_bindings);
}
PyObject* reset_statistics(PyObject*,PyObject*){statistics=Statistics{};Py_RETURN_NONE;}
PyMethodDef methods[]={
    {"make_plan",make_plan,METH_VARARGS,"Compile immutable expected facts only."},
    {"compare",compare,METH_VARARGS,"True fast pass; false/unsupported require full literal fallback."},
    {"build_info",build_info,METH_NOARGS,"API, ABI and compiled source identity."},
    {"binary_identity",binary_identity,METH_NOARGS,"Actual mapped compare image path/base/entry."},
    {"entry_identity",entry_identity,METH_O,"Immutable C method definition, entry address and owner binding."},
    {"statistics",get_statistics,METH_NOARGS,"Observed native comparisons, fallback causes and fact counts."},
    {"reset_statistics",reset_statistics,METH_NOARGS,"Reset diagnostics only; no proof or expected facts changed."},
    {nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_native_guard",nullptr,-1,methods};
}
PyMODINIT_FUNC PyInit__native_guard() {
#if PY_MAJOR_VERSION != 3 || PY_MINOR_VERSION != 11 || !defined(_WIN64) || (!defined(_M_X64) && !defined(__x86_64__))
    PyErr_SetString(PyExc_ImportError,"native guard supports CPython3.11 Windows x64 only");return nullptr;
#else
    if(PyInterpreterState_Get()!=PyInterpreterState_Main()) {
        PyErr_SetString(PyExc_ImportError,"native guard does not support subinterpreters");return nullptr;
    }
    PlanType.tp_name="anyfesolver._native_guard.Plan";PlanType.tp_basicsize=sizeof(PlanObject);
    PlanType.tp_flags=Py_TPFLAGS_DEFAULT|Py_TPFLAGS_IMMUTABLETYPE|Py_TPFLAGS_HAVE_GC;
    PlanType.tp_dealloc=reinterpret_cast<destructor>(plan_dealloc);PlanType.tp_new=nullptr;
    PlanType.tp_traverse=reinterpret_cast<traverseproc>(plan_traverse);PlanType.tp_clear=reinterpret_cast<inquiry>(plan_clear);
    if(PyType_Ready(&PlanType)<0)return nullptr;
    return PyModule_Create(&module);
#endif
}
