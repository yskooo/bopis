const fs = require('fs');
const html = fs.readFileSync('bopis.html', 'utf8');
const scriptMatch = html.match(/<script>(.*?)<\/script>/s);
let code = scriptMatch[1];

// Mock DOM
global.window = { devicePixelRatio: 1, addEventListener: () => {} };
global.location = { hash: '', protocol: 'http:' };
global.history = { replaceState: () => {} }; global.localStorage = { getItem:()=>null, setItem:()=>{} }; global.sessionStorage = { getItem:()=>null, setItem:()=>{} }; global.cancelAnimationFrame=()=>{}; global.requestAnimationFrame=()=>{};
global.document = {
    addEventListener: () => {},
    createElement: () => ({}),
    getElementById: (id) => {
        if (id === 'pareto') return {
            parentElement: { clientWidth: 1000 },
            style: {},
            getContext: () => ({
                setTransform: () => {},
                clearRect: () => {},
                beginPath: () => {},
                moveTo: () => {},
                lineTo: () => {},
                stroke: () => {},
                fillText: () => {},
                arc: () => {},
                fill: () => {}
            })
        };
        return { checked: false, textContent: '', innerHTML: '' };
    }
};

global.RUN_DATA = {
    pareto: {
        points: [
            { config: 'c1', energy_j: 1, tokens_per_s: 2, quality_f1: 0.5, on_front: true },
            { config: 'c2', energy_j: 2, tokens_per_s: 3, quality_f1: 0.6, on_front: true }
        ],
        random_points: []
    },
    reference_point: { energy_j: 3, tokens_per_s: 1, quality_f1: 0.4 }
};

global.setTimeout = () => {};
global.fetch = () => Promise.resolve({ok:true, json:()=>Promise.resolve({})});

try {
    eval(code);
    drawPareto3D();
    console.log("Success");
} catch(e) {
    console.error("Error:", e.stack);
}
