const fs = require('node:fs');
const vm = require('node:vm');
const { lineageNeighbors, artefactLineageNeighbors, lineageTargets } = require('../collectra/gui/lineage.js');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const between = (from, to) => html.slice(html.indexOf(from), html.indexOf(to, html.indexOf(from)));

// sheet(data) -> detect(task) -> crop(data) -> read(task) -> text(data)
//                            \-> hand(data)
const nodes = [
    {id: 'sheet', nodeType: 'data'}, {id: 'detect', nodeType: 'detector'},
    {id: 'crop', nodeType: 'data'}, {id: 'hand', nodeType: 'data'},
    {id: 'read', nodeType: 'llm'}, {id: 'text', nodeType: 'data'},
];
const edges = [
    {source: 'sheet', target: 'detect'}, {source: 'detect', target: 'crop'}, {source: 'detect', target: 'hand'},
    {source: 'crop', target: 'read'}, {source: 'read', target: 'text'},
];

describe('lineageNeighbors', () => {
    it('finds descendants of the same kind, passing through the tasks between them', () => {
        expect(lineageNeighbors('sheet', 'descendants', nodes, edges)).toEqual(['crop', 'hand', 'text']);
    });

    it('finds ancestors of the same kind at any depth', () => {
        expect(lineageNeighbors('text', 'ancestors', nodes, edges)).toEqual(['crop', 'sheet']);
    });

    it('a task sees only tasks', () => {
        expect(lineageNeighbors('read', 'ancestors', nodes, edges)).toEqual(['detect']);
        expect(lineageNeighbors('detect', 'descendants', nodes, edges)).toEqual(['read']);
    });

    it('has none at the ends of the pipeline', () => {
        expect(lineageNeighbors('sheet', 'ancestors', nodes, edges)).toEqual([]);
        expect(lineageNeighbors('text', 'descendants', nodes, edges)).toEqual([]);
        expect(lineageNeighbors('hand', 'descendants', nodes, edges)).toEqual([]);
    });

    it('is empty for a node the pipeline does not have', () => {
        expect(lineageNeighbors('nope', 'ancestors', nodes, edges)).toEqual([]);
        expect(lineageNeighbors(undefined, 'descendants', nodes, edges)).toEqual([]);
    });

    it('skips nodes it is told to, and does not walk through them', () => {
        const skip = (id) => id === 'crop';
        expect(lineageNeighbors('sheet', 'descendants', nodes, edges, skip)).toEqual(['hand']);
    });

    it('lists each node once and survives a cycle', () => {
        const loop = [...edges, {source: 'text', target: 'sheet'}];
        const result = lineageNeighbors('sheet', 'descendants', nodes, loop);
        expect(new Set(result).size).toBe(result.length);
        expect(result).not.toContain('sheet');
    });

    it('copes with no nodes or edges', () => {
        expect(lineageNeighbors('sheet', 'ancestors', [], [])).toEqual([]);
        expect(lineageNeighbors('sheet', 'ancestors', undefined, undefined)).toEqual([]);
    });
});

describe('Ancestors and Descendants do not depend on any panel being open', () => {
    const source = between('function lineageAvailability(', 'function openLineageFor(');

    function page({withGraph}) {
        const context = {
            lineageNeighbors, lineageTargets, pageNodeIds: new Map(), pageNodeParents: null,
            resolveDagLabel: (id) => id.replace(/\d+$/, ''), PAGE_NODE_RE: /^Page/,
            lastDagNodes: nodes, lastDagEdges: edges,
            // The drawn graph of the top DAG panel: gone (that panel was removed).
            cy: withGraph ? {} : null,
        };
        vm.createContext(context);
        vm.runInContext(source, context);
        return context;
    }

    it('says a node has ancestors and descendants even when there is no drawn graph at all', () => {
        const ctx = page({withGraph: false});
        expect(ctx.lineageAvailability('crop1')).toEqual({ancestors: true, descendants: true});
        expect(ctx.lineageAvailability('sheet')).toEqual({ancestors: false, descendants: true});
        expect(ctx.lineageAvailability('text')).toEqual({ancestors: true, descendants: false});
    });

    it('gives the same answer with or without a drawn graph', () => {
        expect(page({withGraph: true}).lineageAvailability('crop')).toEqual(page({withGraph: false}).lineageAvailability('crop'));
    });

    it('says none for no node or an unknown one', () => {
        const ctx = page({withGraph: false});
        expect(ctx.lineageAvailability(null)).toEqual({ancestors: false, descendants: false});
        expect(ctx.lineageAvailability('unknown_label')).toEqual({ancestors: false, descendants: false});
    });

    it('the lists in the popovers come from the same data, never from the drawn graph', () => {
        const render = between('function renderLineage(', '\n        }\n');
        expect(render).toContain('lineageTargets(nodeId, direction');
        expect(render).toContain("targets('ancestors')");
        expect(render).toContain("targets('descendants')");
        expect(render).not.toContain('cy.');
        expect(render).not.toContain('!cy');
        expect(source).not.toContain('cy.');
    });

    it('loads the module', () => {
        expect(html).toContain('<script src="lineage.js"></script>');
    });

    it('the old walk over the drawn graph is gone', () => {
        expect(html).not.toContain('function sameTypeNeighbors(');
        expect(html).not.toContain('LINEAGE_TASK_TYPES');
    });
});

describe('the buttons catch up when the pipeline data arrives', () => {
    it('every panel\'s row is redrawn as soon as the nodes and edges are known', () => {
        const loaded = between('lastDagEdges = edges;', 'initCytoscape(');
        expect(loaded).toContain('Object.values(panelControllers).forEach((c) => c.renderPager());');
    });
});

const instanceParents = new Map([
    ['page', []], ['region-a', ['page']], ['region-b', ['page']],
    ['line-a1', ['region-a']], ['line-a2', ['region-a']], ['line-b1', ['region-b']],
    ['text-a1', ['line-a1']],
]);

it('follows the selected line instance parents, excluding sibling regions and lines', () => {
    expect(artefactLineageNeighbors('line-a1', 'ancestors', instanceParents)).toEqual(['region-a', 'page']);
    expect(artefactLineageNeighbors('text-a1', 'ancestors', instanceParents)).toEqual(['line-a1', 'region-a', 'page']);
    expect(artefactLineageNeighbors('region-a', 'descendants', instanceParents)).toEqual(['line-a1', 'line-a2', 'text-a1']);
});

it('does not expand an instance ancestor category to all of that category instances', () => {
    const ids = new Map([['crop', ['region-a', 'region-b']], ['text', ['line-a1', 'line-b1']]]);
    const resolve = id => id.startsWith('line') ? 'text' : 'crop';
    expect(lineageTargets('line-a1', 'ancestors', nodes, edges, ids, instanceParents, resolve)).toEqual(['region-a', 'page']);
});

it('handles shared ancestors, dangling parents and cycles without including the selected instance', () => {
    const parents = new Map([['a', ['b', 'c']], ['b', ['root']], ['c', ['root', 'missing']], ['root', ['a']]]);
    expect(artefactLineageNeighbors('a', 'ancestors', parents)).toEqual(['b', 'c', 'root']);
    expect(artefactLineageNeighbors('missing', 'ancestors', parents)).toEqual([]);
});

it('renders only the selected instance lineage and updates when the selected instance changes', () => {
    const lists = {parentsList: {}, childrenList: {}};
    const context = {
        document: {getElementById: id => lists[id]},
        PAGE_NODE_RE: /^Page/, activeNodeIds: null, lastDagNodes: nodes, lastDagEdges: edges,
        pageNodeIds: new Map(), pageNodeParents: instanceParents, lineageTargets,
        resolveDagLabel: id => id, truncateId: id => id, escapeHtml: id => id,
        repositionLineagePopovers() {},
    };
    vm.createContext(context);
    vm.runInContext(between('function renderLineage(', '\n        }\n') + '\n        }', context);
    context.renderLineage('line-a1');
    expect(lists.parentsList.innerHTML).toContain('region-a');
    expect(lists.parentsList.innerHTML).toContain('page');
    expect(lists.parentsList.innerHTML).not.toContain('region-b');
    expect(lists.parentsList.innerHTML).not.toContain('line-a2');
    context.renderLineage('line-b1');
    expect(lists.parentsList.innerHTML).toContain('region-b');
    expect(lists.parentsList.innerHTML).not.toContain('region-a');
});

// ── per-artefact lineage must be ready as soon as a page opens ──────────────────────────

describe('the page data behind the lineage lists', () => {
    const source = between('function applyPageNodeData(', '\n        }\n') + '\n        }';

    function load(initial = {}) {
        const context = {
            pageNodeTypes: new Map(), pageCropLabels: new Set(), pageNodeIds: new Map(),
            pageNodeParents: null, ...initial,
        };
        vm.createContext(context);
        vm.runInContext(source, context);
        return context;
    }

    const result = {
        success: true,
        node_types: {crop: 'collectra.ImageCrop'},
        crop_labels: ['crop'],
        node_ids: {crop: ['region-a', 'region-b']},
        node_parents: {page: [], 'region-a': ['page'], 'region-b': ['page'], 'line-a1': ['region-a']},
    };

    it('records the types, crop labels, ids and each artefact\'s own parents together', () => {
        const ctx = load();
        ctx.applyPageNodeData(result);
        expect(ctx.pageNodeTypes.get('crop')).toBe('collectra.ImageCrop');
        expect(ctx.pageCropLabels.has('crop')).toBe(true);
        expect(ctx.pageNodeIds.get('crop')).toEqual(['region-a', 'region-b']);
        expect(ctx.pageNodeParents.get('line-a1')).toEqual(['region-a']);
    });

    it('replaces the previous page\'s parents rather than keeping them', () => {
        const ctx = load({pageNodeParents: new Map([['old-page-node', ['x']]])});
        ctx.applyPageNodeData(result);
        expect(ctx.pageNodeParents.has('old-page-node')).toBe(false);
    });

    it('forgets the previous page when the new page has no parent data or fails', () => {
        const ctx = load({pageNodeParents: new Map([['old', []]]), pageNodeIds: new Map([['old', ['o']]])});
        ctx.applyPageNodeData({success: false, error: 'nope'});
        expect(ctx.pageNodeParents).toBe(null);
        expect(ctx.pageNodeIds.size).toBe(0);
        expect(ctx.pageNodeTypes.size).toBe(0);
        expect(ctx.pageCropLabels.size).toBe(0);
    });

    it('gives only the selected artefact\'s own lineage right after the page opens', () => {
        const ctx = load();
        ctx.applyPageNodeData(result);
        const targets = (id, direction) =>
            lineageTargets(id, direction, nodes, edges, ctx.pageNodeIds, ctx.pageNodeParents, (x) => x);
        expect(targets('region-a', 'descendants')).toEqual(['line-a1']);
        expect(targets('region-b', 'descendants')).toEqual([]);   // not the sibling's lines
        expect(targets('line-a1', 'ancestors')).toEqual(['region-a', 'page']);
    });

    it('opening a page and refreshing it both use it, so neither can be missed', () => {
        const opening = between('const activeResult = await window.pywebview.api.get_active_node_ids(index);', 'populateTabs(lastDagNodes);');
        const refreshing = between('async function refreshActiveNodeIds', 'populateTabs(lastDagNodes);');
        expect(opening).toContain('applyPageNodeData(activeResult)');
        expect(refreshing).toContain('applyPageNodeData(activeResult)');
        expect(opening).not.toContain('pageNodeIds = new Map');
        expect(refreshing).not.toContain('pageNodeIds = new Map');
    });
});
