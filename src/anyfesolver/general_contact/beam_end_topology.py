"""Physical classical-beam ends, including unselected material-node junctions."""


def physical_beam_ends(model,cells,beam_types):
    incidence={}
    for element in model.mesh.elements.values():
        if isinstance(element,beam_types):
            for node in set(element.node_ids):
                incidence[int(node)]=incidence.get(int(node),0)+1
    ends={}
    for cell in cells:
        for local,sign in ((0,-1),(len(cell.node_ids)-1,1)):
            node=int(cell.node_ids[local])
            if incidence.get(node)==1:ends[node]=(cell,local,sign)
    return ends
