
        (function () {
            try {
                const heroFx = document.getElementById('hero-fx');
                const canvas = document.getElementById('hero-canvas');
                const slotEl = document.getElementById('brand-slot');
                const sideCanvas = document.getElementById('sidebar-canvas');
                if (!heroFx || !canvas) throw new Error('hero-fx markup missing');
                const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
                const ctx = canvas.getContext('2d');
                const sideCtx = sideCanvas && sideCanvas.getContext('2d');
                const dpr = Math.min(window.devicePixelRatio || 1, 2);
                const TAU = Math.PI * 2;
                let w = 0, h = 0, raf = null, dismissed = false, paused = false, angle = 0;
                let mode = 'full', morph = null;               // full -> morph -> mini
                let last = performance.now();
                let sideW = 0, sideH = 0;

                function fitEl(el, c, cw, ch) {
                    el.width = Math.floor(cw * dpr); el.height = Math.floor(ch * dpr);
                    el.style.width = cw + 'px'; el.style.height = ch + 'px';
                    c.setTransform(dpr, 0, 0, dpr, 0, 0);
                }
                function fit(cw, ch) {
                    w = cw; h = ch;
                    fitEl(canvas, ctx, w, h);
                }
                function fitSidebar() {
                    if (!sideCanvas || !sideCtx) return;
                    const s = (sideCanvas.parentElement && Math.round(sideCanvas.parentElement.getBoundingClientRect().width)) || 40;
                    sideW = sideH = s;
                    fitEl(sideCanvas, sideCtx, s, s);
                }
                function size() { if (mode === 'full') fit(window.innerWidth, window.innerHeight); }
                size();
                fitSidebar();
                window.addEventListener('resize', size);
                window.addEventListener('resize', fitSidebar);

                // Cyan #00E5FF core -> deep navy #001133 rim.
                const STOPS = [[0, 229, 255], [0, 150, 230], [0, 64, 150], [0, 17, 51]];
                function ramp(t) {
                    const pos = Math.max(0, Math.min(0.999, t)) * (STOPS.length - 1);
                    const i = Math.floor(pos), f = pos - i, a = STOPS[i], b = STOPS[i + 1];
                    return 'rgb(' + a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(',') + ')';
                }

                const ARMS = 3, DOTS = 130, TURNS = 3.2, ROT = 0.17;   // ROT = rad/s
                const SPRING = 0.075, DAMP = 0.86, TRAIL = 6, MINI = 0.4, DUR = 1.25;
                const ease = x => x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
                const dots = [];
                for (let arm = 0; arm < ARMS; arm++) {
                    for (let i = 0; i < DOTS; i++) {
                        const t = i / (DOTS - 1);
                        dots.push({
                            arm: (arm / ARMS) * TAU, t, color: ramp(t), x: 0, y: 0, vx: 0, vy: 0,
                            ix: 0, iy: 0, ox: 0, oy: 0, seeded: false, trail: []
                        });
                    }
                }

                const mouse = { x: 0, y: 0, sx: 0, sy: 0, px: 0, py: 0, speed: 0, active: false, seeded: false };
                function onMove(e) {
                    mouse.x = e.clientX; mouse.y = e.clientY;
                    if (!mouse.seeded) { mouse.sx = mouse.px = mouse.x; mouse.sy = mouse.py = mouse.y; mouse.seeded = true; }
                    mouse.active = true;
                }
                function onLeave() { mouse.active = false; }
                window.addEventListener('pointermove', onMove, { passive: true });
                document.documentElement.addEventListener('mouseleave', onLeave);

                function toMini() {                             // hand the canvas over to the header slot
                    mode = 'mini';
                    const s = (slotEl && Math.round(slotEl.getBoundingClientRect().width)) || 44;
                    canvas.classList.add('mini');
                    if (slotEl) slotEl.appendChild(canvas);
                    fit(s, s);
                    heroFx.remove();
                }

                function paintSpiral(c, cw, ch, t, dt, opts) {
                    const sizeMul = opts.sizeMul;
                    const physics = !!opts.physics;
                    const e = opts.e || 0;
                    const cx = opts.cx, cy = opts.cy, maxR = opts.maxR;
                    const breathe = 1 + 0.03 * Math.sin(t * 1.3);

                    if (physics && mouse.active) {
                        const k = 1 - Math.pow(0.8, dt);
                        mouse.sx += (mouse.x - mouse.sx) * k; mouse.sy += (mouse.y - mouse.sy) * k;
                        const sp = Math.hypot(mouse.sx - mouse.px, mouse.sy - mouse.py) / dt;
                        mouse.speed += (sp - mouse.speed) * 0.2;
                        mouse.px = mouse.sx; mouse.py = mouse.sy;
                    } else if (physics) { mouse.speed *= 0.9; }
                    const R = Math.max(140, Math.min(window.innerWidth, window.innerHeight) * 0.17), R2 = R * R;
                    const boost = 2.4 + Math.min(mouse.speed, 30) * 0.12;
                    const damp = Math.pow(DAMP, dt);
                    c.clearRect(0, 0, cw, ch);

                    for (let n = 0; n < dots.length; n++) {
                        const d = dots[n];
                        const phase = t * 2.0 - d.t * 9;
                        const r = d.t * maxR * breathe * (1 + 0.018 * Math.sin(phase));
                        const th = d.arm + angle + d.t * TURNS * TAU;
                        const ix = cx + Math.cos(th) * r, iy = cy + Math.sin(th) * r;
                        let x = ix, y = iy, push = 0, vx = 0, vy = 0;

                        if (physics) {
                            d.ix = ix; d.iy = iy;
                            if (!d.seeded) { d.x = ix; d.y = iy; d.seeded = true; }
                            if (mouse.active) {
                                const dx = d.x - mouse.sx, dy = d.y - mouse.sy, d2 = dx * dx + dy * dy;
                                if (d2 < R2 && d2 > 0.01) {
                                    const dist = Math.sqrt(d2), ux = dx / dist, uy = dy / dist;
                                    const f = (1 - dist / R) * (1 - dist / R) * boost;
                                    d.vx += (ux - uy * 0.35) * f * dt;
                                    d.vy += (uy + ux * 0.35) * f * dt;
                                }
                            }
                            d.vx += (ix - d.x) * SPRING * dt; d.vy += (iy - d.y) * SPRING * dt;
                            d.vx *= damp; d.vy *= damp;
                            d.x += d.vx * dt; d.y += d.vy * dt;
                            push = Math.min(1, Math.max(0, Math.hypot(d.x - ix, d.y - iy) - 3) / 45);
                            x = d.x; y = d.y; vx = d.vx; vy = d.vy;
                        } else if (opts.followMain) {
                            const k = mode === 'morph' ? 1 - e : 0;
                            x = ix + d.ox * k; y = iy + d.oy * k;
                            d.x = x; d.y = y; d.vx = d.vy = 0;
                        }

                        const odd = n & 1;
                        const am = odd ? (opts.thinOdds || 0) : 1;
                        const tr = physics ? d.trail : null;
                        if (tr) {
                            if (push > 0.06) { tr.push(x, y); if (tr.length > TRAIL * 2) { tr.shift(); tr.shift(); } }
                            else if (tr.length) { tr.shift(); tr.shift(); }
                        }
                        if (am <= 0.01) continue;

                        const size = Math.max(0.5, (0.9 + d.t * 2.6) * (1 + 0.22 * Math.sin(phase)) * (1 + push * 0.5) * sizeMul);
                        const alpha = (0.55 + d.t * 0.4) * am;
                        c.fillStyle = d.color;
                        const cnt = tr ? tr.length / 2 : 0;
                        for (let k = 0; k < cnt - 1; k++) {
                            const a = (k + 1) / cnt;
                            c.globalAlpha = alpha * 0.35 * a;
                            c.beginPath(); c.arc(tr[2 * k], tr[2 * k + 1], size * (0.35 + 0.5 * a), 0, TAU); c.fill();
                        }
                        c.globalAlpha = alpha;
                        c.beginPath();
                        if (push > 0.12) {
                            c.save(); c.translate(x, y); c.rotate(Math.atan2(vy, vx));
                            c.ellipse(0, 0, size * (1 + push * 3.2), size * (1 + push * 0.4), 0, 0, TAU);
                            c.fill(); c.restore();
                        } else {
                            c.arc(x, y, size, 0, TAU); c.fill();
                        }
                    }
                    c.globalAlpha = 1;
                }

                function paintSidebar(t) {
                    if (!sideCtx || !sideW) return;
                    paintSpiral(sideCtx, sideW, sideH, t, 1, {
                        cx: sideW / 2, cy: sideH / 2, maxR: sideW / 2 - 2, sizeMul: MINI, thinOdds: 0
                    });
                }

                function step(t, dt) {
                    angle += ROT * dt / 60;
                    let cx = 0, cy = 0, maxR = 0, sizeMul = 1, e = 0;
                    if (mode === 'full') { cx = w / 2; cy = h / 2; maxR = Math.min(w, h) * 0.52; }
                    if (mode === 'morph') {
                        const p = Math.min(1, (t - morph.t0) / DUR);
                        e = ease(p);
                        if (p >= 1) { toMini(); }
                        else {
                            const r = slotEl.getBoundingClientRect();
                            const tx = r.left + r.width / 2, ty = r.top + r.height / 2, tR = Math.max(6, r.width / 2 - 2);
                            cx = morph.cx0 + (tx - morph.cx0) * e; cy = morph.cy0 + (ty - morph.cy0) * e;
                            maxR = morph.R0 + (tR - morph.R0) * e; sizeMul = 1 + (MINI - 1) * e;
                        }
                    }
                    if (mode === 'mini') { cx = cy = w / 2; maxR = w / 2 - 2; sizeMul = MINI; e = 1; }
                    const thinOdds = mode === 'full' ? 1 : mode === 'morph' ? 1 - e : 0;
                    paintSpiral(ctx, w, h, t, dt, {
                        cx, cy, maxR, sizeMul, e,
                        physics: mode === 'full',
                        followMain: mode !== 'full',
                        thinOdds
                    });
                    paintSidebar(t);
                }

                function frame(ts) {
                    raf = requestAnimationFrame(frame);
                    const dt = Math.min(2.5, Math.max(0.2, (ts - last) / 16.667));
                    last = ts;
                    if (paused) return;
                    const headerGone = mode === 'mini' && canvas.offsetParent === null;
                    const sideGone = !sideCanvas || sideCanvas.offsetParent === null;
                    if (headerGone && sideGone) return;
                    if (headerGone) {
                        angle += ROT * dt / 60;
                        paintSidebar(ts / 1000);
                        return;
                    }
                    step(ts / 1000, dt);
                }

                function setLit() {
                    if (dismissed) return;
                    heroFx.classList.add('fx-lit');
                    document.body.classList.add('hero-lit');
                }

                function dismiss() {
                    if (dismissed) return;
                    dismissed = true;
                    document.body.classList.remove('landing');
                    document.body.classList.add('chatting', 'hero-lit');
                    heroFx.classList.add('fx-lit', 'fx-dismissed');
                    window.removeEventListener('resize', size);
                    window.removeEventListener('pointermove', onMove);
                    document.documentElement.removeEventListener('mouseleave', onLeave);
                    for (const d of dots) { d.ox = d.x - d.ix; d.oy = d.y - d.iy; d.trail.length = 0; }
                    const r = slotEl && slotEl.getBoundingClientRect();
                    const now = performance.now() / 1000;
                    if (!r || r.width === 0 || reduceMotion) {    // no visible header (or reduced motion): land instantly
                        mode = 'mini'; toMini(); step(now, 1);
                    } else {
                        mode = 'morph';
                        morph = { t0: now, cx0: w / 2, cy0: h / 2 };
                    }
                    if (mode === 'morph') morph.R0 = Math.min(w, h) * 0.52;
                }
                window.__dismissHeroFx = dismiss;

                document.addEventListener('visibilitychange', function () {
                    paused = document.hidden; last = performance.now();
                });

                step(performance.now() / 1000, 1);              // first frame immediately, never blank
                if (reduceMotion) { setLit(); return; }
                raf = requestAnimationFrame(frame);
                setTimeout(setLit, 700);                        // brief black, then slow cross-fade to white
            } catch (err) {
                console.error('hero-fx failed, falling back to plain chat:', err);
                document.body.classList.remove('landing');
                document.body.classList.add('chatting', 'hero-lit');
                const fx = document.getElementById('hero-fx'); if (fx) fx.remove();
                const sideSlot = document.getElementById('sidebar-brand-slot');
                if (sideSlot) sideSlot.outerHTML = '<div class="avatar">B</div>';
            }
        })();
        let gpDemoInitialized = false;
        function toggleGPDemo() {
            const act = document.getElementById('convergence-actual-wrapper');
            const dem = document.getElementById('convergence-demo-wrapper');
            if (act.style.display === 'none') {
                act.style.display = 'block';
                dem.style.display = 'none';
            } else {
                act.style.display = 'none';
                dem.style.display = 'block';
                if (!gpDemoInitialized) initGPDemo();
            }
        }
        function initGPDemo() {
            gpDemoInitialized = true;
            const f=x=>0.55*Math.sin(7*x)+0.35*Math.sin(15*x+0.5)+0.3;
            const L=0.09,S=0.6,NZ=1e-4;
            const kf=(a,b)=>S*S*Math.exp(-((a-b)**2)/(2*L*L));
            let pts,it,msg;
            function inv(M){const n=M.length;const A=M.map((r,i)=>[...r,...Array.from({length:n},(_,j)=>i==j?1:0)]);
            for(let c=0;c<n;c++){let p=c;for(let r=c+1;r<n;r++)if(Math.abs(A[r][c])>Math.abs(A[p][c]))p=r;[A[c],A[p]]=[A[p],A[c]];const d=A[c][c];for(let j=0;j<2*n;j++)A[c][j]/=d;for(let r=0;r<n;r++)if(r!=c){const m=A[r][c];if(m)for(let j=0;j<2*n;j++)A[r][j]-=m*A[c][j];}}
            return A.map(r=>r.slice(n));}
            function erf(x){const s=Math.sign(x);x=Math.abs(x);const t=1/(1+0.3275911*x);return s*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x));}
            const Phi=z=>0.5*(1+erf(z/Math.SQRT2)),phi=z=>Math.exp(-z*z/2)/Math.sqrt(2*Math.PI);
            function model(){
            const m=pts.reduce((a,p)=>a+p.y,0)/pts.length;
            const K=pts.map((a,i)=>pts.map((b,j)=>kf(a.x,b.x)+(i==j?NZ:0)));
            const Ki=inv(K);const yc=pts.map(p=>p.y-m);
            const al=Ki.map(r=>r.reduce((s,v,j)=>s+v*yc[j],0));
            const best=Math.max(...pts.map(p=>p.y));const G=[];
            for(let i=0;i<=150;i++){const x=i/150;const ks=pts.map(p=>kf(x,p.x));
            const mu=m+ks.reduce((s,v,j)=>s+v*al[j],0);
            const t=Ki.map(r=>r.reduce((s,v,j)=>s+v*ks[j],0));
            const sd=Math.sqrt(Math.max(1e-9,S*S-ks.reduce((s,v,j)=>s+v*t[j],0)));
            const d=mu-best-0.01,z=d/sd;G.push({x,mu,sd,ei:Math.max(0,d*Phi(z)+sd*phi(z))});}
            return{G,best};}
            const X=x=>50+x*600,cl=v=>Math.max(20,Math.min(250,v)),Y=y=>cl(250-(y+1)/2.6*230);
            function draw(){
            const{G,best}=model();let nx=G[0];G.forEach(g=>{if(g.ei>nx.ei)nx=g;});
            const em=Math.max(...G.map(g=>g.ei))||1;
            let h='';
            h+='<line x1="50" y1="250" x2="650" y2="250" stroke="#E2E8F0" stroke-width="0.5"/><line x1="50" y1="370" x2="650" y2="370" stroke="#E2E8F0" stroke-width="0.5"/>';
            h+='<text x="50" y="14" font-size="12" fill="#718096">Score (higher is better)</text>';
            h+='<text x="50" y="278" font-size="12" fill="#718096">Expected improvement (where to test next)</text>';
            h+='<text x="350" y="410" font-size="12" fill="#718096" text-anchor="middle">Configuration setting (e.g. quantization level, GPU layers)</text>';
            const up=G.map(g=>X(g.x)+','+Y(g.mu+2*g.sd)).join(' '),lo=G.slice().reverse().map(g=>X(g.x)+','+Y(g.mu-2*g.sd)).join(' ');
            h+='<polygon points="'+up+' '+lo+'" fill="#7F77DD" opacity="0.2"/>';
            if(document.getElementById('gp-tr').checked){let d='';for(let i=0;i<=150;i++){const x=i/150;d+=(i?'L':'M')+X(x)+' '+Y(f(x));}h+='<path d="'+d+'" fill="none" stroke="#A0AEC0" stroke-width="1" stroke-dasharray="4 4"/>';}
            h+='<path d="'+G.map((g,i)=>(i?'L':'M')+X(g.x)+' '+Y(g.mu)).join('')+'" fill="none" stroke="#7F77DD" stroke-width="2"/>';
            h+='<polygon points="50,370 '+G.map(g=>X(g.x)+','+(370-g.ei/em*70)).join(' ')+' 650,370" fill="#D85A30" opacity="0.3"/>';
            h+='<line x1="'+X(nx.x)+'" y1="20" x2="'+X(nx.x)+'" y2="370" stroke="#D85A30" stroke-width="1" stroke-dasharray="4 3"/>';
            h+='<text x="'+Math.min(X(nx.x)+6,600)+'" y="34" font-size="12" fill="#718096">next test</text>';
            pts.forEach(p=>{const b=p.y===best;h+='<circle cx="'+X(p.x)+'" cy="'+Y(p.y)+'" r="'+(b?7:5)+'" fill="'+(b?'#EF9F27':'#1D9E75')+'"/>';});
            document.getElementById('gp-demo-svg').innerHTML=h;
            window._nx=nx.x;
            document.getElementById('gp-demo-info').innerHTML='Evaluations: '+pts.length+' &middot; Best score: '+best.toFixed(3)+' &middot; Next test at x = '+nx.x.toFixed(2)+'<br>'+msg;}
            function step(){const x=window._nx,y=f(x);pts.push({x,y});it++;msg='Iteration '+it+': EI picked x = '+x.toFixed(2)+', the real test scored '+y.toFixed(3)+', and the GP was refit with it.';draw();}
            function reset(){pts=[0.12,0.5,0.88].map(x=>({x,y:f(x)}));it=0;msg='Started with 3 seed configurations. The GP guess is wide where nothing has been tested.';draw();}
            document.getElementById('gp-b1').onclick=step;
            document.getElementById('gp-b5').onclick=()=>{for(let i=0;i<5;i++){step();}};
            document.getElementById('gp-br').onclick=reset;
            document.getElementById('gp-tr').onchange=draw;
            reset();
        }
    