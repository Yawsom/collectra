const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const start = html.indexOf('async function loadCollectraFolderInner(index)');
const end = html.indexOf('// Collapse the sidebar', start);
const source = html.slice(start, end);

function pageContext({carry = null, activeTab = null} = {}) {
    const item = {classList: {add: vi.fn(), remove: vi.fn()}, scrollIntoView: vi.fn()};
    const elements = {folderInfo: {}, yamlPath: {}};
    const bottom = {
        panel: {getActiveTab: () => activeTab},
        reconcileTabs: vi.fn(), renderTabStripView: vi.fn(),
        showActiveTabContent: vi.fn(), renderPageArtefact: vi.fn(), openOrActivateTab: vi.fn(),
    };
    const other = {reconcileTabs: vi.fn(), renderTabStripView: vi.fn(), showActiveTabContent: vi.fn()};
    const context = {
        document: {querySelectorAll: () => [item], getElementById: id => elements[id]},
        window: {pywebview: {api: {
            load_collectra_folder: vi.fn().mockResolvedValue({success: true, folder_path: '/page', image_path: '/page/image.png', yaml_path: '/page/results.yaml'}),
            get_label_statistics: vi.fn().mockResolvedValue({success: true, label_counts: {}}),
            get_active_node_ids: vi.fn().mockResolvedValue({success: true, active_ids: ['image', 'crop'], node_ids: {crop: ['crop-001']}}),
        }}},
        loadYaml: vi.fn(), updateNavButtons: vi.fn(), populateTabs: vi.fn(), setActive: vi.fn(),
        panelControllers: {bottom, other}, activeNodeIds: new Set(['image', 'crop']),
        resolveDagLabel: id => id, pendingCarryLabel: carry, PAGE_NODE_RE: /^Page/,
        cy: {nodes: () => ({roots: () => [{id: () => 'image'}]})}, lastDagNodes: [],
        showResult: vi.fn(), formatLabelCounts: vi.fn(), rerunArtefactSearch: vi.fn(),
    };
    return {context, bottom, other, item};
}

// The page data is applied by one shared function; run the real one.
const applyStart = html.indexOf('function applyPageNodeData(');
const applySource = html.slice(applyStart, html.indexOf('\n        }\n', applyStart)) + '\n        }\n';
const run = (context) => vm.runInNewContext(`${applySource}\n${source}\nloadCollectraFolderInner(0)`, context);

it.each([null, 'crop'])('renders only the final selected artefact when carrying %s', async carry => {
    const {context, bottom, other, item} = pageContext({carry});
    await run(context);
    expect(bottom.renderPageArtefact).not.toHaveBeenCalled();
    expect(bottom.showActiveTabContent).not.toHaveBeenCalled();
    expect(bottom.openOrActivateTab).toHaveBeenCalledTimes(1);
    expect(bottom.openOrActivateTab).toHaveBeenCalledWith(carry ? 'crop-001' : 'image');
    expect(other.showActiveTabContent).toHaveBeenCalledTimes(1);
    // Label tabs follow the new page: kept while the label has an instance there, on its first one.
    expect(bottom.reconcileTabs).toHaveBeenCalledWith({preferFirst: true});
    expect(other.reconcileTabs).toHaveBeenCalledWith({preferFirst: true});
    expect(context.showResult).not.toHaveBeenCalled();
    expect(context.rerunArtefactSearch).toHaveBeenCalledTimes(1);
    expect(item.scrollIntoView).toHaveBeenCalledWith({block: 'nearest', inline: 'nearest'});
});

describe('switching page keeps the tabs the user has open', () => {
    it('shows the active tab again instead of opening the carried field or the root', async () => {
        const {context, bottom} = pageContext({carry: 'crop', activeTab: {id: 't1', kind: 'node', groupKey: 'crop'}});
        await run(context);
        expect(bottom.openOrActivateTab).not.toHaveBeenCalled();
        expect(bottom.showActiveTabContent).toHaveBeenCalledTimes(1);
    });

    it('does the same when the active tab is the DAG', async () => {
        const {context, bottom} = pageContext({activeTab: {id: 'd', kind: 'dag'}});
        await run(context);
        expect(bottom.openOrActivateTab).not.toHaveBeenCalled();
        expect(bottom.showActiveTabContent).toHaveBeenCalledTimes(1);
    });

    it('opens the root only when the panel has no tabs at all', async () => {
        const {context, bottom} = pageContext({});
        await run(context);
        expect(bottom.openOrActivateTab).toHaveBeenCalledWith('image');
        expect(bottom.showActiveTabContent).not.toHaveBeenCalled();
    });
});
