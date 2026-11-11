"""Native guard-local authority matching; no proof survives an invocation.

Expected tuples are interned using identities only. Live function metadata is
checked anew by every guard. Globals/class facts remain their caller's checks.
"""
import types


def intern_facts(records):
    """Share only identical captured predicates, retaining cyclic leaf meaning."""
    interned, visited = {}, {}
    def node(fact):
        if id(fact) in visited:
            return visited[id(fact)]
        if not fact:
            key, made = ('empty',), ()
        elif len(fact) == 1:
            child = fact[0]
            if isinstance(child, tuple):
                child = node(child)
                key, made = ('descriptor', id(child)), (child,)
            else:
                key, made = ('identity', id(child)), fact
        elif len(fact) == 5:
            member, code, defaults, keywords, closures = fact
            entries = tuple((value,node(expected),generation) for value,expected,generation in closures)
            key = ('function',id(member),id(code),id(defaults),
                   None if keywords is None else tuple((id(k),id(v)) for k,v in keywords),
                   tuple((id(v),id(a),g) for v,a,g in entries))
            made = (member,code,defaults,keywords,entries)
        else:
            children = tuple(node(child) for child in fact)
            key, made = ('property',tuple(id(child) for child in children)), children
        result = interned.setdefault(key,made)
        visited[id(fact)] = result
        return result
    return tuple((member,node(authority),bindings) for member,authority,bindings in records)


def matches(member, authority, memo):
    """Literal boundary predicate, with memo scoped to one native guard only."""
    key = (id(member),id(authority))
    if key in memo:
        previous_member,previous_authority,result=memo[key]
        if previous_member is member and previous_authority is authority:
            return result
    if type(member) is types.FunctionType:
        if len(authority) == 1:
            result = member is authority[0]
        elif not authority or member is not authority[0] or member.__code__ is not authority[1] or member.__defaults__ is not authority[2]:
            result = False
        else:
            closure = member.__closure__ or ()
            if len(closure) != len(authority[4]):
                result = False
            else:
                try:
                    invalid = any((type(cell.cell_contents) is not int if generation_counter else
                        cell.cell_contents is not value or not matches(value,captured,memo))
                        for cell,(value,captured,generation_counter) in zip(closure,authority[4]))
                except ValueError:
                    invalid = True
                expected, current = authority[3], member.__kwdefaults__
                result = (not invalid and (current is None if expected is None else
                    type(current) is dict and all(type(k) is str for k in current) and len(current)==len(expected)
                    and all(current.get(k,_MISSING) is v for k,v in expected)))
    elif type(member) is property:
        result = len(authority)==3 and all(matches(f,a,memo) for f,a in zip((member.fget,member.fset,member.fdel),authority))
    elif type(member) in (staticmethod,classmethod):
        result = len(authority)==1 and matches(member.__func__,authority[0],memo)
    else:
        result = not authority
    memo[key] = (member,authority,result)
    return result


_MISSING = object()
