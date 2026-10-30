const html = require('./displaySources.js');
const between = (from, to) => html.slice(html.indexOf(from), html.indexOf(to, html.indexOf(from)));

describe('no hover cards over a dialog', () => {
    it('hides every hover card while any dialog is open', () => {
        const start = html.indexOf('body:has(.modal-overlay.visible)');
        expect(start).toBeGreaterThan(-1);
        const rule = html.slice(start, html.indexOf('}', start));
        for (const card of ['#pageBoxTooltip', '#tabTooltip', '#dagTooltip', '#dagTabTooltip', '.dag-tab-tooltip']) {
            expect(rule, card).toContain(card);
        }
        expect(rule).toContain('display: none !important;');
    });

    it('does not start showing the box hover while a dialog is open', () => {
        const enter = between("this.pageAnno.on('mouseEnterAnnotation'", "this.pageAnno.on('mouseLeaveAnnotation'");
        expect(enter).toContain("document.querySelector('.modal-overlay.visible')");
    });

    it('drops the box hover as soon as the right-click menu or the delete dialog opens', () => {
        expect(html).toContain('function hideBoxTooltip()');
        expect(between('function showAnnotationContextMenu(', 'function showFolderContextMenu')).toContain('hideBoxTooltip();');
        expect(between('function confirmDelete(', 'async function deleteNodeWithConfirm(')).toContain('hideBoxTooltip();');
    });

    it('puts no hover text on the rows of the delete list', () => {
        const list = between('function confirmDelete(', 'async function deleteNodeWithConfirm(');
        expect(list).not.toContain('.title =');
        expect(list).not.toContain('title=');
    });
});

describe('hover cards in the DAG tabs of split panels', () => {
    const vm = require('node:vm');
    const source = between('function hideAllHoverCards(', 'document.addEventListener(\'mousedown\'');

    it('are cleared too, though they have no id', () => {
        const shown = (id) => ({ id, style: { display: 'block' } });
        const byId = { pageBoxTooltip: shown('a'), tabTooltip: shown('b'), dagTooltip: shown('c'), dagTabTooltip: shown('d') };
        const splitPanelCards = [shown('e'), shown('f')];
        const context = {
            cancelHoverHide: () => {},
            document: {
                getElementById: (id) => byId[id] || null,
                querySelectorAll: (sel) => (sel === '.dag-tab-tooltip' ? splitPanelCards : []),
            },
        };
        vm.createContext(context);
        vm.runInContext(source, context);
        context.hideAllHoverCards();
        for (const card of [...Object.values(byId), ...splitPanelCards]) expect(card.style.display).toBe('none');
    });
});


describe('folder right-click menu', () => {
    const menu = () => between('function showFolderContextMenu(', 'function getDagColors(');

    it('offers Copy Path beside Reveal in Finder', () => {
        const source = menu();
        expect(source).toContain('data-action="reveal">Reveal in Finder');
        expect(source).toContain('data-action="copy-path">Copy Path');
    });

    it('copies the specimen folder path through the Python bridge', () => {
        const source = menu();
        expect(source).toContain('copy_to_clipboard(');
        expect(source).toMatch(/folder\.yaml_path[^;]*replace\(/);
    });
});
