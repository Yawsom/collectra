// What the page does when its own record of a page's artefacts is missing or out of step
// with what the backend knows (it must not leave tabs nameless or buttons greyed out).
const vm = require('node:vm');
const { nodeActionButtons, pagerState } = require('../collectra/gui/instancePager.js');
const { lineageNeighbors } = require('../collectra/gui/lineage.js');
const html = require('./displaySources.js');
const between = (from, to) => html.slice(html.indexOf(from), html.indexOf(to, html.indexOf(from)));

describe('opening an artefact with a pipeline label from the backend (search hits)', () => {
    const source = between('async openOrActivateTab(nodeId', '\n            }\n') + '\n            }\n';

    function setup({ pageNodeIds = new Map(), existing = null }) {
        const opened = [];
        const controller = {
            panel: {
                tabs: existing ? [existing] : [],
                openOrActivateGroup: (spec) => { opened.push(spec); return spec; },
            },
            renderTabStripView: vi.fn(), showActiveTabContent: vi.fn().mockResolvedValue(), 
        };
        const context = {
            pageNodeIds, nextTabId: 1, resolveDagLabel: (id) => id,        // cannot resolve: the page record is empty
            humanizeLabel: (l) => l.split('_').map((w) => w[0].toUpperCase() + w.slice(1)).join(' '),
            onPanelTabsChanged: vi.fn(),
        };
        vm.createContext(context);
        vm.runInContext(`var holder = { ${source} };`, context);
        return { opened, controller, open: (...a) => context.holder.openOrActivateTab.apply(controller, a) };
    }
    const hitId = 'locality-18925dcf-6676-4990-8e88-6a5c0a7db284';

    it('names the tab after the label, not after the id, even when the page record is empty', async () => {
        const { opened, open } = setup({});
        await open(hitId, 'locality');
        expect(opened[0].groupKey).toBe('locality');
        expect(opened[0].label).toBe('Locality');
        expect(opened[0].nodeId).toBe(hitId);
    });

    it('shows the hit itself when that label\'s tab is already open on another instance', async () => {
        const { opened, open } = setup({ existing: { kind: 'node', groupKey: 'locality', nodeId: 'locality-other' } });
        await open(hitId, 'locality');
        expect(opened[0].nodeId).toBe(hitId);
    });

    it('still works out the label itself when none is given (a click on the DAG or a list)', async () => {
        const { opened, open } = setup({});
        await open('some_label');
        expect(opened[0].groupKey).toBe('some_label');
    });

    it('search passes each hit\'s label on', () => {
        const step = html.slice(html.indexOf('async function stepArtefactSearch('), html.indexOf('// ── END ARTEFACT SEARCH'));
        expect(step).toContain('openOrActivateTab(matches[searchState.index].id, matches[searchState.index].label)');
    });
});

describe('Ancestors and Descendants follow the tab\'s selected instance', () => {
    it('the row\'s buttons use its selected instance rather than its whole category', () => {
        const source = between('renderPager() {', 'reconcileTabs(options)');
        const asked = [];
        const pagerEl = { hidden: true, children: [], replaceChildren(...k) { this.children = k; } };
        const mk = () => ({ setAttribute() {}, addEventListener() {}, style: {} });
        const controller = {
            el: { headerType: { parentElement: { querySelector: () => pagerEl } } },
            panel: { getActiveTab: () => ({ kind: 'node', nodeId: 'locality-18925dcf', groupKey: 'locality' }) },
            instancesForTab: () => ['locality-18925dcf'],
        };
        const context = {
            pagerState, nodeActionButtons, NODE_ACTION_ICONS: {},
            document: { createElement: mk },
            lineageAvailability: (what) => { asked.push(what); return { ancestors: true, descendants: false }; },
        };
        vm.createContext(context);
        vm.runInContext(`var holder = { ${source} };`, context);
        context.holder.renderPager.call(controller);
        expect(asked).toEqual(['locality-18925dcf']);
    });

    it('so does the list in the popovers', () => {
        expect(html).toContain('renderLineage(tab.nodeId || tab.groupKey);');
    });

    it('a label the pipeline knows has lineage even when no instance of it is on the page record', () => {
        const nodes = [{ id: 'sheet', nodeType: 'data' }, { id: 'det', nodeType: 'detector' }, { id: 'locality', nodeType: 'data' }];
        const edges = [{ source: 'sheet', target: 'det' }, { source: 'det', target: 'locality' }];
        expect(lineageNeighbors('locality', 'ancestors', nodes, edges)).toEqual(['sheet']);
    });
});

describe('Artefacts and Tasks do not wait for the DAG to be drawn', () => {
    const source = between('async function loadAndRenderDag() {', '\n        }\n') + '\n        }\n';

    function setup({ initThrows }) {
        const els = {};
        const el = (id) => (els[id] = els[id] || { id, disabled: true, style: {}, dataset: {}, textContent: '' });
        const context = {
            apiReady: true, lastDagNodes: [], lastDagEdges: [], panelControllers: { top: { renderPager: vi.fn() } },
            document: { getElementById: el }, console: { error() {} },
            window: { pywebview: { api: { get_pipeline_graph: async () => ({ success: true, ext: 'grapto', nodes: [{ id: 'a' }], edges: [] }) } } },
            initCytoscape: () => { if (initThrows) throw new Error('could not draw'); },
        };
        vm.createContext(context);
        vm.runInContext(source, context);
        return { els, context, run: () => context.loadAndRenderDag() };
    }

    it('are enabled once the nodes and edges are known, even if drawing the graph then fails', async () => {
        const { els, run } = setup({ initThrows: true });
        await run();
        expect(els.artefactsToolbarBtn.disabled).toBe(false);
        expect(els.tasksToolbarBtn.disabled).toBe(false);
    });

    it('are enabled when everything works too', async () => {
        const { els, run } = setup({ initThrows: false });
        await run();
        expect(els.artefactsToolbarBtn.disabled).toBe(false);
    });

    it('the rows\' buttons are redrawn from the new data, and one failing row cannot stop the load', async () => {
        const { context, run, els } = setup({ initThrows: false });
        context.panelControllers = { a: { renderPager: () => { throw new Error('bad row'); } }, b: { renderPager: vi.fn() } };
        await run();
        expect(els.artefactsToolbarBtn.disabled).toBe(false);
    });

    it('is enabled before anything that can fail', () => {
        const fn = between('lastDagEdges = edges;', 'initCytoscape(nodes, edges);');
        expect(fn).toContain("document.getElementById('artefactsToolbarBtn').disabled = false;");
        expect(fn).toContain("document.getElementById('tasksToolbarBtn').disabled = false;");
        expect(fn).toMatch(/try \{[\s\S]*renderPager\(\)[\s\S]*\} catch/);
    });
});

describe('the page record of artefacts is kept up to date, and never thrown away', () => {
    const source = between('async function refreshActiveNodeIds() {', '\n        }\n') + '\n        }\n';
    const applySource = between('function applyPageNodeData(', '\n        }\n') + '\n        }\n';

    function setup(answer) {
        const context = {
            currentFolderIndex: 2,
            pageNodeTypes: new Map([['old', 'collectra.Text']]),
            pageCropLabels: new Set(['old']),
            pageNodeIds: new Map([['old', ['old-1']]]),
            pageNodeParents: null,
            lastDagNodes: [{ id: 'old' }],
            populateTabs: vi.fn(), setActive: vi.fn(), console: { error() {} },
            window: { pywebview: { api: { get_active_node_ids: vi.fn(answer) } } },
        };
        vm.createContext(context);
        vm.runInContext(applySource + source, context);
        return context;
    }

    it('replaces the record with the backend\'s new one, and redraws the lists and the DAG\'s active nodes', async () => {
        const ctx = setup(async () => ({ success: true, node_ids: { locality: ['locality-1', 'locality-2'] }, node_types: { locality: 'collectra.Text' }, crop_labels: ['locality'], active_ids: ['locality'] }));
        await ctx.refreshActiveNodeIds();
        expect([...ctx.pageNodeIds.keys()]).toEqual(['locality']);
        expect(ctx.pageNodeIds.get('locality')).toEqual(['locality-1', 'locality-2']);
        expect(ctx.populateTabs).toHaveBeenCalled();
        expect(ctx.setActive).toHaveBeenCalledWith(['locality']);
    });

    it('keeps the record it had when the backend reports a failure, instead of emptying it', async () => {
        const ctx = setup(async () => ({ success: false, error: 'busy' }));
        await ctx.refreshActiveNodeIds();
        expect(ctx.pageNodeIds.get('old')).toEqual(['old-1']);
        expect(ctx.pageNodeTypes.has('old')).toBe(true);
        expect(ctx.setActive).not.toHaveBeenCalled();
    });

    it('keeps the record, and does not throw, when the call itself fails', async () => {
        const ctx = setup(async () => { throw new Error('bridge down'); });
        await expect(ctx.refreshActiveNodeIds()).resolves.toBeUndefined();
        expect(ctx.pageNodeIds.get('old')).toEqual(['old-1']);
    });

    it('does nothing before a page is open', async () => {
        const ctx = setup(async () => ({ success: true, node_ids: {} }));
        ctx.currentFolderIndex = -1;
        await ctx.refreshActiveNodeIds();
        expect(ctx.window.pywebview.api.get_active_node_ids).not.toHaveBeenCalled();
    });
});

describe('a search is repeated whenever what it searches may have changed', () => {
    it('after a rename', () => {
        expect(between('async function confirmRenameName(', '\n        }\n')).toContain('rerunArtefactSearch();');
    });

    it('after a new crop is added', () => {
        const create = between("const result = await window.pywebview.api.create_annotation(", 'hideLabelPopup();\n            } catch');
        expect(create).toContain('rerunArtefactSearch();');
    });

    it('after text is saved', () => {
        const save = between('this.textSave = (view ? view.editable : true)', 'mountEditor(view.text, isTei);');
        expect(save).toContain("rerunArtefactSearch();");
    });
});

describe('when the pipeline data arrives', () => {
    const arrived = between('lastDagEdges = edges;', 'initCytoscape(nodes, edges);');

    it('tabs that are not for a pipeline label are dropped, and the layout is saved without them', () => {
        expect(arrived).toContain('removeNodeTabsNotIn(');
        expect(arrived).toContain('renderTabStripView()');
    });

    it('a DAG tab drawn before the data was known is drawn again with it', () => {
        const refresh = between('function refreshDagTabs(', '\n        }\n');
        expect(refresh).toContain('destroyDagTab()');
        expect(refresh).toContain('renderDagTab()');
        expect(arrived).toContain('refreshDagTabs();');
    });

    it('refreshDagTabs redraws only the DAG tabs that are showing', () => {
        const source = between('function refreshDagTabs(', '\n        }\n') + '\n        }\n';
        const make = (kind, hasEls) => {
            const c = { dagTabEls: hasEls ? {} : null, destroyDagTab: vi.fn(), renderDagTab: vi.fn(), panel: { getActiveTab: () => (kind ? { kind } : null) } };
            return c;
        };
        const showing = make('dag', true), notShowing = make('node', true), top = make('dag', false), empty = make(null, true);
        const context = { panelControllers: { showing, notShowing, top, empty } };
        vm.createContext(context);
        vm.runInContext(source, context);
        context.refreshDagTabs();
        expect(showing.destroyDagTab).toHaveBeenCalled();
        expect(showing.renderDagTab).toHaveBeenCalled();
        expect(notShowing.destroyDagTab).toHaveBeenCalled(); // stale either way
        expect(notShowing.renderDagTab).not.toHaveBeenCalled(); // drawn when its tab is next opened
        expect(top.renderDagTab).not.toHaveBeenCalled(); // the original top panel draws through initCytoscape
        expect(empty.renderDagTab).not.toHaveBeenCalled();
    });
});
