(() => {
  'use strict';
  const D=JSON.parse(document.getElementById('trace-data').textContent);
  const $=id=>document.getElementById(id),F=D.frames,ga=D.mode==='ga',sa=D.mode==='sa';
  const fmt=n=>Number(n).toLocaleString('en-US',{maximumFractionDigits:2});
  const safe=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const line=(x1,y1,x2,y2,color='#dce5ed',extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" ${extra}/>`;
  const txt=(x,y,s,extra='')=>`<text x="${x}" y="${y}" font-size="11" font-family="sans-serif" fill="#526379" ${extra}>${safe(s)}</text>`;
  const path=(points,color,extra='')=>`<polyline points="${points.map(p=>p.join(',')).join(' ')}" fill="none" stroke="${color}" stroke-width="2.3" ${extra}/>`;
  let index=0,timer=null;
  $('name').textContent=D.name;$('seed').textContent=`Seed ${D.seed} · grid "${D.grid_name}"`;
  $('scrub').max=F.length-1;$('route').value='current';
  $('settings').textContent=JSON.stringify(D.settings,null,2);
  if(ga){$('costLabel').textContent='Population best cost';$('curveLegend').textContent='━ Population best';}
  $('caption').textContent='Clues are bold and never move. Shaded cells hold a digit repeated in their row, column or box; '+
    'outlined cells changed since the previous frame. The cost is your objective; the shaded count is a fixed display rule. '+
    'Frames are sampled; evaluation counts are exact.';
  const values=D.events.flatMap(e=>[e.cost,e.best]);
  const vmin=Math.min(0,...values),vmax=Math.max(1,...values)*1.05;
  const cx=e=>52+435*(e-1)/Math.max(1,D.total-1),cy=v=>152-130*(v-vmin)/Math.max(1e-9,vmax-vmin);
  const temps=F.map(f=>f.temperature||0),maxT=Math.max(1e-12,...temps);
  function conflicts(g){
    const bad=Array(81).fill(false);
    const unit=cells=>{const seen={};cells.forEach(i=>{const v=g[i];if(v){(seen[v]=seen[v]||[]).push(i);}});
      Object.values(seen).forEach(list=>{if(list.length>1)list.forEach(i=>bad[i]=true);});};
    for(let r=0;r<9;r++)unit([...Array(9).keys()].map(c=>r*9+c));
    for(let c=0;c<9;c++)unit([...Array(9).keys()].map(r=>r*9+c));
    for(let b=0;b<9;b++){const r0=3*Math.floor(b/3),c0=3*(b%3);unit([...Array(9).keys()].map(k=>(r0+Math.floor(k/3))*9+c0+k%3));}
    return bad;
  }
  function shown(f){
    const route=$('route').value;
    if(route==='best')return {grid:f.best_grid,title:'Best archived grid',cost:f.best};
    if(route==='candidate'){
      if(f.candidate)return {grid:f.candidate,title:'Candidate grid',cost:f.candidate_cost};
      return {grid:f.grid,title:'No candidate recorded · current grid',cost:f.cost};
    }
    return {grid:f.grid,title:ga?'Population champion':'Current grid',cost:f.cost};
  }
  function drawGrid(f){
    const view=shown(f),g=view.grid;
    if(!Array.isArray(g)||g.length!==81||g.some(v=>!Number.isInteger(v)||v<1||v>9))throw Error('Invalid grid in trace.');
    for(let i=0;i<81;i++)if(D.puzzle[i]&&g[i]!==D.puzzle[i])throw Error('A clue changed in the trace.');
    const prev=index>0?shown(F[index-1]).grid:null,bad=conflicts(g),size=40,o=15;
    let s='<rect x="0" y="0" width="390" height="390" rx="9" fill="#f7fafc"/>';
    for(let i=0;i<81;i++){
      const r=Math.floor(i/9),c=i%9,x=o+c*size,y=o+r*size;
      s+=`<rect x="${x}" y="${y}" width="${size}" height="${size}" fill="${bad[i]?'#f6d2cc':'#ffffff'}"/>`;
      if(prev&&prev[i]!==g[i])s+=`<rect x="${x+3}" y="${y+3}" width="${size-6}" height="${size-6}" fill="none" stroke="#d26532" stroke-width="2.5" rx="3"/>`;
      const clue=D.puzzle[i]>0;
      s+=`<text x="${x+size/2}" y="${y+size/2+6}" text-anchor="middle" font-size="18" font-family="sans-serif" font-weight="${clue?700:400}" fill="${clue?'#182b45':'#087f8c'}">${g[i]}</text>`;
    }
    for(let k=0;k<=9;k++){const w=k%3===0?2.6:.7;s+=line(o,o+k*size,o+9*size,o+k*size,'#182b45',`stroke-width="${w}"`)+line(o+k*size,o,o+k*size,o+9*size,'#182b45',`stroke-width="${w}"`);}
    $('map').innerHTML=s;
    const n=bad.filter(Boolean).length;
    $('mapLabel').textContent=`${view.title} · cost ${fmt(view.cost)}`;
    $('topology').textContent=n?`${n} cells hold a repeated digit · ${D.puzzle.filter(v=>v>0).length} clues kept`:'No repeated digit: this grid is solved';
  }
  function drawCurve(f){
    let s='';
    for(let k=0;k<4;k++){let v=vmin+(vmax-vmin)*k/3,y=cy(v);s+=line(52,y,490,y)+txt(44,y+4,fmt(v),'text-anchor="end"');}
    for(let k=0;k<3;k++){let e=1+(D.total-1)*k/2;s+=txt(cx(e),173,Math.round(e),'text-anchor="middle"');}
    s+=txt(270,193,'Objective evaluations','text-anchor="middle"');
    const events=D.events.filter(e=>e.evaluations<=f.evaluations);
    if(events.length){
      s+=path(events.map(e=>[cx(e.evaluations),cy(e.cost)]),'#087f8c');
      s+=path(events.map(e=>[cx(e.evaluations),cy(e.best)]),'#7059b8','stroke-dasharray="5 4"');
    }
    s+=line(cx(f.evaluations),15,cx(f.evaluations),154,'#b1becc','stroke-dasharray="3 4"');
    s+=`<circle cx="${cx(f.evaluations)}" cy="${cy(f.cost)}" r="4" fill="#087f8c"/>`;
    $('curve').innerHTML=s;
  }
  function drawDiagnostic(f){
    let s='';const st=f.stats||{};
    if(ga&&f.population_costs){
      const costs=f.population_costs,lo=Math.min(...costs),hi=Math.max(...costs),bins=12,counts=Array(bins).fill(0);
      costs.forEach(v=>counts[Math.min(bins-1,Math.floor((v-lo)/Math.max(1,hi-lo)*bins))]++);
      const max=Math.max(...counts);
      counts.forEach((c,i)=>{const h=c/max*49;s+=`<rect x="${35+i*35}" y="${69-h}" width="29" height="${h}" fill="#7059b8" opacity=".7"/>`;});
      s+=txt(35,12,'Population cost distribution · count')+txt(35,86,fmt(lo))+txt(451,86,fmt(hi),'text-anchor="end"');
      s+=txt(35,108,`Population diversity recorded in the trace: ${fmt((f.diversity||0)*100)}%`);
    }else if(sa&&f.temperature!==null){
      const temp=f.temperature||0;
      s+=txt(15,19,`Temperature: ${fmt(temp)}`)+txt(495,19,`T / maximum recorded T = ${fmt(temp/maxT*100)}%`,'text-anchor="end"');
      s+='<rect x="15" y="30" width="480" height="10" rx="5" fill="#e1e8ef"/>';
      s+=`<rect x="15" y="30" width="${480*temp/maxT}" height="10" rx="5" fill="#d26532"/>`;
      s+=txt(15,66,`Worse proposals accepted so far: ${st.worse_accepted||0} / ${st.worse||0}`);
      s+=txt(15,91,'Compare the temperature with the typical cost change of one move.');
    }else if(sa){
      s+=txt(15,24,`Worse proposals accepted so far: ${st.worse_accepted||0} / ${st.worse||0}`);
      s+=txt(15,52,`Accepted proposals: ${st.accepted||0} / ${st.decisions||0}`);
      s+=txt(15,80,'Every rise of the current curve is an accepted worse move.');
    }else{
      s+=txt(15,24,`Accepted steps: ${st.accepted||0} / ${st.decisions||0} decisions · sideways moves: ${st.sideways||0}`);
      s+=txt(15,52,`Random restarts so far: ${st.restarts||0}`);
      s+=txt(15,80,'A plateau: many neighbours with exactly the same cost.');
    }
    $('diagnostic').innerHTML=s;
  }
  function render(){
    const f=F[index];$('cost').textContent=fmt(f.cost);$('best').textContent=fmt(f.best);$('evaluation').textContent=`${f.evaluations} / ${D.total}`;
    $('scrub').value=index;$('frame').textContent=`Frame ${index+1} / ${F.length}${ga&&f.generation!==null?' · generation '+f.generation:''}${index===F.length-1?` · end of the run · ${D.steps} steps`:''}`;
    let title=f.accepted===null?'Recorded state':f.accepted?'Accepted':'Rejected';
    let detail=f.note||(f.delta===null?'Initial state.':`Δ = ${fmt(f.delta)} · candidate cost ${fmt(f.candidate_cost)}`);
    if(sa&&f.probability!==null)detail+=` · acceptance probability ${f.probability.toFixed(3)}`;
    if(ga){title=f.generation===null?'Population state':`Generation ${f.generation}`;detail=f.note||'Recorded after the population update.';}
    $('decision').innerHTML=`<b>${safe(title)}</b>${safe(detail)}`;
    drawGrid(f);drawCurve(f);drawDiagnostic(f);
  }
  function stop(){if(timer)clearInterval(timer);timer=null;$('play').textContent='Play';$('play').setAttribute('aria-label','Play animation');}
  function play(){if(index===F.length-1)index=0;$('play').textContent='Pause';$('play').setAttribute('aria-label','Pause animation');render();timer=setInterval(()=>{if(index>=F.length-1){stop();return;}index++;render();},Number($('speed').value));}
  $('play').onclick=()=>timer?stop():play();
  $('back').onclick=()=>{stop();index=Math.max(0,index-1);render();};
  $('forward').onclick=()=>{stop();index=Math.min(F.length-1,index+1);render();};
  $('scrub').oninput=()=>{stop();index=Number($('scrub').value);render();};
  $('speed').onchange=()=>{if(timer){stop();play();}};
  $('route').onchange=()=>{stop();render();};
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  render();
})();
