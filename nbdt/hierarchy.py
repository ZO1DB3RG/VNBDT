from nbdt.utils import DATASETS, METHODS, Colors, fwd
from nbdt.graph import build_minimal_wordnet_graph, build_random_graph, \
    prune_single_successor_nodes, write_graph, get_wnids, generate_fname, \
    get_parser, get_wnids_from_dataset, get_directory, get_graph_path_from_args, \
    augment_graph, get_depth, build_induced_graph, read_graph, get_leaves, \
    get_roots, synset_to_wnid, wnid_to_name, get_root
from nbdt import data
from networkx.readwrite.json_graph import adjacency_data
from pathlib import Path
import os
import json
import torchvision
import base64
from io import BytesIO
from collections import defaultdict


############
# GENERATE #
############


def print_graph_stats(G, name):
    num_children = [len(succ) for succ in G.succ]
    print('[{}] \t Nodes: {} \t Depth: {} \t Max Children: {}'.format(
        name,
        len(G.nodes),
        get_depth(G),
        max(num_children)))


def assert_all_wnids_in_graph(G, wnids):
    assert all(wnid.strip() in G.nodes for wnid in wnids), [
        wnid for wnid in wnids if wnid not in G.nodes
    ]


def generate_hierarchy(
        dataset, method, seed=1, branching_factor=2, extra=0,
        no_prune=False, fname='', path='', single_path=False,
        induced_linkage='ward', induced_affinity='euclidean',
        checkpoint=None, arch=None, model=None, classes = [], **kwargs):
    wnids = get_wnids_from_dataset(dataset)#简单的为每一个类别对应的wordnet ID进行对应

    if method == 'wordnet':
        G = build_minimal_wordnet_graph(wnids, single_path)
    elif method == 'random':
        G = build_random_graph(wnids, seed=seed, branching_factor=branching_factor)
    elif method == 'induced':
        G, index_to_tensor= build_induced_graph(wnids,
            dataset=dataset,
            checkpoint=checkpoint,
            model=arch, #???
            linkage=induced_linkage,
            affinity=induced_affinity,
            branching_factor=branching_factor,
            classes = classes,
            state_dict=model.state_dict() if model is not None else None) #会转到graph.py中的该函数
    else:
        raise NotImplementedError(f'Method "{method}" not yet handled.')
    print_graph_stats(G, 'matched')
    assert_all_wnids_in_graph(G, wnids)

    if not no_prune:
        G = prune_single_successor_nodes(G)
        print_graph_stats(G, 'pruned')
        assert_all_wnids_in_graph(G, wnids)

    if extra > 0:
        G, n_extra, n_imaginary = augment_graph(G, extra, True)
        print(f'[extra] \t Extras: {n_extra} \t Imaginary: {n_imaginary}')
        print_graph_stats(G, 'extra')
        assert_all_wnids_in_graph(G, wnids)

    path = get_graph_path_from_args(
        dataset=dataset,
        method=method,
        seed=seed,
        branching_factor=branching_factor,
        extra=extra,
        no_prune=no_prune,
        fname=fname,
        path=path,
        single_path=single_path,
        induced_linkage=induced_linkage,
        induced_affinity=induced_affinity,
        checkpoint=checkpoint,
        arch=arch)
    write_graph(G, path)

    Colors.green('==> Wrote tree to {}'.format(path))


########
# TEST #
########


def get_seen_wnids(wnid_set, nodes):
    leaves_seen = set()
    for leaf in nodes:
        if leaf in wnid_set:
            wnid_set.remove(leaf)
        if leaf in leaves_seen:
            pass
        leaves_seen.add(leaf)
    return leaves_seen


def match_wnid_leaves(wnids, G, tree_name):
    wnid_set = set()
    for wnid in wnids:
        wnid_set.add(wnid.strip())

    leaves_seen = get_seen_wnids(wnid_set, get_leaves(G))
    return leaves_seen, wnid_set


def match_wnid_nodes(wnids, G, tree_name):
    wnid_set = {wnid.strip() for wnid in wnids}
    leaves_seen = get_seen_wnids(wnid_set, G.nodes)

    return leaves_seen, wnid_set


def print_stats(leaves_seen, wnid_set, tree_name, node_type):
    print(f"[{tree_name}] \t {node_type}: {len(leaves_seen)} \t WNIDs missing from {node_type}: {len(wnid_set)}")
    if len(wnid_set):
        Colors.red(f"==> Warning: WNIDs in wnid.txt are missing from {tree_name} {node_type}")


def test_hierarchy(args):
    wnids = get_wnids_from_dataset(args.dataset)
    path = get_graph_path_from_args(**vars(args))
    print('==> Reading from {}'.format(path))

    G = read_graph(path)

    G_name = Path(path).stem

    leaves_seen, wnid_set1 = match_wnid_leaves(wnids, G, G_name)
    print_stats(leaves_seen, wnid_set1, G_name, 'leaves')

    leaves_seen, wnid_set2 = match_wnid_nodes(wnids, G, G_name)
    print_stats(leaves_seen, wnid_set2, G_name, 'nodes')

    num_roots = len(list(get_roots(G)))
    if num_roots == 1:
        Colors.green('Found just 1 root.')
    else:
        Colors.red(f'Found {num_roots} roots. Should be only 1.')

    if len(wnid_set1) == len(wnid_set2) == 0 and num_roots == 1:
        Colors.green("==> All checks pass!")
    else:
        Colors.red('==> Test failed')


#######
# VIS #
#######


def set_dot_notation(node, key, value):
    """
    >>> a = {}
    >>> set_dot_notation(a, 'above.href', 'hello')
    >>> a['above']['href']
    'hello'
    """
    curr = last = node
    key_part = key
    if '.' in key:
        for key_part in key.split('.'):
            last = curr
            curr[key_part] = node.get(key_part, {})
            curr = curr[key_part]
    last[key_part] = value


def build_tree(G, root,
        parent='null',
        color_info=(),
        force_labels_left=(),
        include_leaf_images=False,
        dataset=None,
        image_resize_factor=1,
        include_fake_sublabels=False,
        include_fake_labels=False,
        node_to_conf={}):
    """
    :param color_info dict[str, dict]: mapping from node labels or IDs to color
                                       information. This is by default just a
                                       key called 'color'
    """
    children = [
        build_tree(G, child, root,
            color_info=color_info,
            force_labels_left=force_labels_left,
            include_leaf_images=include_leaf_images,
            dataset=dataset,
            image_resize_factor=image_resize_factor,
            include_fake_sublabels=include_fake_sublabels,
            include_fake_labels=include_fake_labels,
            node_to_conf=node_to_conf)
        for child in G.succ[root]]
    _node = G.nodes[root]
    label = _node.get('label', '')
    sublabel = root

    if root.startswith('f') and label.startswith('(') and not include_fake_labels:
        label = ''

    if root.startswith('f') and not include_fake_sublabels:  # WARNING: hacky, ignores fake wnids -- this will have to be changed lol
        sublabel = ''

    node = {
        'sublabel': sublabel,
        'label': label,
        'parent': parent,
        'children': children,
        'alt': _node.get('alt', ', '.join(map(wnid_to_name, get_leaves(G, root=root)))),
        'id': root
    }

    if label in color_info:
        node.update(color_info[label])

    if root in color_info:
        node.update(color_info[root])

    if label in force_labels_left:
        node['force_text_on_left'] = True

    is_leaf = len(children) == 0
    if include_leaf_images and is_leaf:
        try:
            image = get_class_image_from_dataset(dataset, label)
        except UserWarning as e:
            print(e)
            return node
        base64_encode = image_to_base64_encode(image, format="jpeg")
        image_href = f"data:image/jpeg;base64,{base64_encode.decode('utf-8')}"
        image_height, image_width = image.size
        node['image'] = {
            'href': image_href,
            'width': image_width * image_resize_factor,
            'height': image_height * image_resize_factor
        }

    for key, value in node_to_conf[root].items():
        set_dot_notation(node, key, value)
    return node


def build_graph(G):
    return {
        'nodes': [{
            'name': wnid,
            'label': G.nodes[wnid].get('label', ''),
            'id': wnid
        } for wnid in G.nodes],
        'links': [{
            'source': u,
            'target': v
        } for u, v in G.edges]
    }


def get_class_image_from_dataset(dataset, candidate):
    """Returns image for given class `candidate`. Image is PIL."""
    if isinstance(candidate, int):
        candidate = dataset.classes[candidate]
    for sample, label in dataset:
        intersection = compare_wnids(dataset.classes[label], candidate)
        if label == candidate or intersection:
            return sample
    raise UserWarning(f'No samples with label {candidate} found.')


def compare_wnids(label1, label2):
    from nltk.corpus import wordnet as wn  # entire script should not depend on wordnet
    synsets1 = wn.synsets(label1, pos=wn.NOUN)
    synsets2 = wn.synsets(label2, pos=wn.NOUN)
    wnids1 = set(map(synset_to_wnid, synsets1))
    wnids2 = set(map(synset_to_wnid, synsets2))
    return wnids1.intersection(wnids2)


def image_to_base64_encode(image, format="jpeg"):
    """Converts PIL image to base64 encoding, ready for use as data uri."""
    buffered = BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue())


def generate_vis(path_template, data, fname, zoom=2, straight_lines=True,
        show_sublabels=False, height=750, dark=False, margin_top=20,
        above_dy=325, y_node_sep=170, hide=[], _print=False, out_dir='.',
        scale=1, colormap='colormap_annotated.png', below_dy=475, root_y='null',
        width=1000, margin_left=250):
    with open(path_template) as f:
        html = f.read() \
        .replace(
            "CONFIG_MARGIN_LEFT",
            str(margin_left)) \
        .replace(
            "CONFIG_VIS_WIDTH",
            str(width)) \
        .replace(
            "CONFIG_SCALE",
            str(scale)) \
        .replace(
            "CONFIG_PRINT",
            str(_print).lower()) \
        .replace(
            "CONFIG_HIDE",
            str(hide)) \
        .replace(
            "CONFIG_Y_NODE_SEP",
            str(y_node_sep)) \
        .replace(
            "CONFIG_ABOVE_DY",
            str(above_dy)) \
        .replace(
            "CONFIG_BELOW_DY",
            str(below_dy)) \
        .replace(
            "CONFIG_TREE_DATA",
            json.dumps([data])) \
        .replace(
            "CONFIG_ZOOM",
            str(zoom)) \
        .replace(
            "CONFIG_STRAIGHT_LINES",
            str(straight_lines).lower()) \
        .replace(
            "CONFIG_SHOW_SUBLABELS",
            str(show_sublabels).lower()) \
        .replace(
            "CONFIG_TITLE",
            fname) \
        .replace(
            "CONFIG_VIS_HEIGHT",
            str(height)) \
        .replace(
            "CONFIG_BG_COLOR",
            "#111111" if dark else "#FFFFFF") \
        .replace(
            "CONFIG_TEXT_COLOR",
            '#FFFFFF' if dark else '#000000') \
        .replace(
            "CONFIG_TEXT_RECT_COLOR",
            "rgba(17,17,17,0.8)" if dark else "rgba(255,255,255,1)") \
        .replace(
            "CONFIG_MARGIN_TOP",
            str(margin_top)) \
        .replace(
            "CONFIG_ROOT_Y",
            str(root_y)) \
        .replace(
            "CONFIG_COLORMAP",
            f'''<img src="{colormap}" style="
        position: absolute;
        top: 40px;
        left: 80px;
        height: 250px;
        border: 4px solid #ccc;">''' if isinstance(colormap, str) and os.path.exists(colormap) else ''
        )

    os.makedirs(out_dir, exist_ok=True)
    path_html = f'{out_dir}/{fname}.html'
    with open(path_html, 'w') as f:
        f.write(html)

    Colors.green('==> Wrote HTML to {}'.format(path_html))


def get_color_info(G, color, color_leaves, color_path_to=None, color_nodes=()):
    """Mapping from node to color information."""
    nodes = {}
    leaves = list(get_leaves(G))
    if color_leaves:
        for leaf in leaves:
            nodes[leaf] = {'color': color, 'highlighted': True}

    for (id, node) in G.nodes.items():
        if node.get('label', '') in color_nodes or id in color_nodes:
            nodes[id] = {'color': color, 'highlighted': True}

    root = get_root(G)
    target = None
    for leaf in leaves:
        node = G.nodes[leaf]
        if node.get('label', '') == color_path_to or leaf == color_path_to:
            target = leaf
            break

    if target is not None:
        for node in G.nodes:
            nodes[node] = {'color': '#cccccc', 'color_incident_edge': True, 'highlighted': False}

        while target != root:
            nodes[target] = {'color': color, 'color_incident_edge': True, 'highlighted': True}
            view = G.pred[target]
            target = list(view.keys())[0]
        nodes[root] = {'color': color, 'highlighted': True}
    return nodes


def generate_vis_fname(vis_color_path_to=None, vis_out_fname=None, **kwargs):
    fname = vis_out_fname
    if fname is None:
        fname = generate_fname(**kwargs).replace('graph-', f'{kwargs["dataset"]}-', 1)
    if vis_color_path_to is not None:
        fname += '-' + vis_color_path_to
    return fname


def generate_node_conf(node_conf):
    node_to_conf = defaultdict(lambda: {})
    if not node_conf:
        return node_to_conf

    for node, key, value in node_conf:
        if value.isdigit():
            value = int(value)
        node_to_conf[node][key] = value
    return node_to_conf


def generate_hierarchy_vis(args): #也许要在这个函数上做出调整
    path = get_graph_path_from_args(**vars(args))
    print('==> Reading from {}'.format(path))

    G = read_graph(path)

    roots = list(get_roots(G))
    num_roots = len(roots)
    root = args.vis_root or next(get_roots(G))

    assert root in G, f'Node {root} is not a valid node. Nodes: {G.nodes}'

    dataset = None
    if args.dataset and args.vis_leaf_images:
        cls = getattr(data, args.dataset)
        dataset = cls(root='./data', train=False, download=True)

    color_info = get_color_info(
        G,
        args.color,
        color_leaves=not args.vis_no_color_leaves,
        color_path_to=args.vis_color_path_to,
        color_nodes=args.vis_color_nodes or ())

    node_to_conf = generate_node_conf(args.vis_node_conf)

    tree = build_tree(G, root,
        color_info=color_info,
        force_labels_left=args.vis_force_labels_left or [],
        dataset=dataset,
        include_leaf_images=args.vis_leaf_images,
        image_resize_factor=args.vis_image_resize_factor,
        include_fake_sublabels=args.vis_fake_sublabels,
        node_to_conf=node_to_conf)
    graph = build_graph(G)

    if num_roots > 1:
        Colors.red(f'Found {num_roots} roots! Should be only 1: {roots}')
    else:
        print(f'Found just {num_roots} root.')

    fname = generate_vis_fname(**vars(args))
    parent = Path(fwd()).parent
    generate_vis(
        str(parent / 'nbdt/templates/tree-template.html'), tree, fname,
        zoom=args.vis_zoom,
        straight_lines=not args.vis_curved,
        show_sublabels=args.vis_sublabels,
        height=args.vis_height,
        width=args.vis_width,
        dark=args.vis_dark,
        margin_top=args.vis_margin_top,
        margin_left=args.vis_margin_left,
        hide=args.vis_hide or [],
        above_dy=args.vis_above_dy,
        below_dy=args.vis_below_dy,
        scale=args.vis_scale,
        root_y=args.vis_root_y,
        colormap=args.vis_colormap)
