(() => {
  'use strict';
  const D=JSON.parse(document.getElementById('trace-data').textContent);
  const $=id=>document.getElementById(id),F=D.frames,ga=D.mode==='ga',sa=D.mode==='sa',demo=D.mode==='demo';
  const fmt=n=>Number(n).toLocaleString('en-US',{maximumFractionDigits:1});
  const safe=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const line=(x1,y1,x2,y2,color='#dce5ed',extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" ${extra}/>`;
  const txt=(x,y,s,extra='')=>`<text x="${x}" y="${y}" font-size="11" font-family="sans-serif" fill="#526379" ${extra}>${safe(s)}</text>`;
  const path=(points,color,extra='')=>`<polyline points="${points.map(p=>p.join(',')).join(' ')}" fill="none" stroke="${color}" stroke-width="2.3" ${extra}/>`;
  let index=0,timer=null;
  $('name').textContent=D.name;$('seed').textContent=`Seed ${D.seed} · ${D.metric}`;
  $('scrub').max=F.length-1;$('layout').value=D.layout||'geometry';$('route').value='current';$('focus').value='all';
  $('focus').innerHTML='<option value="all">All cities</option>'+D.names.map((name,i)=>`<option value="${i}">${i} · ${safe(name)}</option>`).join('');
  $('cities').textContent=D.names.map((n,i)=>`${i} ${n}`).join('  ·  ');
  $('settings').textContent=JSON.stringify(D.settings,null,2);
  if(ga){$('costLabel').textContent='Population best cost';$('curveLegend').textContent='━ Population best';}
  $('caption').textContent='One tour is drawn at a time. Circular view shows visiting order; costs always use the original coordinates. '+
    'Geometric crossings do not create graph connections. Inspect a city to highlight only its two incident edges. '+
    (demo?'This guided example replays hand-authored moves and decisions, not a search result; your runs use the same controls.':'Tour frames are sampled; evaluation counts are exact.');
  const xs=D.xy.map(p=>p[0]),ys=D.xy.map(p=>p[1]);
  const minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys);
  const scale=Math.min(430/Math.max(1,maxx-minx),320/Math.max(1,maxy-miny));
  const ox=(510-(maxx-minx)*scale)/2,oy=(395-(maxy-miny)*scale)/2;
  const geometric=D.xy.map(p=>[ox+(p[0]-minx)*scale,395-oy-(p[1]-miny)*scale]);
  const values=D.events.flatMap(e=>[e.cost,e.best]);
  const vmin=values.reduce((a,b)=>Math.min(a,b),Infinity)*.94,vmax=values.reduce((a,b)=>Math.max(a,b),-Infinity)*1.03;
  const cx=e=>52+435*(e-1)/Math.max(1,D.total-1),cy=v=>152-130*(v-vmin)/Math.max(1,vmax-vmin);
  const rawMaxT=D.events.map(e=>e.temperature||0).reduce((a,b)=>Math.max(a,b),0);
  const maxT=rawMaxT>0?rawMaxT:1;
  function drawMap(f){
    let tour=f.tour,title=ga?'Population champion':'Current tour',cost=f.cost;
    if($('route').value==='best'){tour=f.best_tour;title='Best archived tour';cost=f.best;}
    if($('route').value==='candidate'){
      if(f.candidate){tour=f.candidate;title='Candidate tour';cost=f.candidate_cost;}
      else title='No candidate recorded · current tour';
    }
    const n=D.names.length,degree=Array(n).fill(0),edges=tour.map((v,i)=>[v,tour[(i+1)%n]]);
    edges.forEach(([a,b])=>{degree[a]++;degree[b]++;});
    if(tour.length!==n||new Set(tour).size!==n||tour.some(v=>!Number.isInteger(v)||v<0||v>=n)||degree.some(v=>v!==2))throw Error('Invalid cycle in trace.');
    const focus=$('focus').value==='all'?null:Number($('focus').value);
    if(focus===null)$('topology').textContent=`Valid single cycle · ${n} cities · ${n} edges · degree 2 at every city`;
    else {
      const at=tour.indexOf(focus),prev=tour[(at+n-1)%n],next=tour[(at+1)%n];
      $('topology').textContent=`${D.names[focus]} (${focus}) · degree 2 · neighbors: ${D.names[prev]} (${prev}) and ${D.names[next]} (${next})`;
    }
    const circle=$('layout').value==='circle';let coords=geometric;
    if(circle){coords=Array(n);tour.forEach((city,i)=>{const a=-Math.PI/2+2*Math.PI*i/n;coords[city]=[255+155*Math.cos(a),198+155*Math.sin(a)];});}
    let s='<rect x="0" y="0" width="510" height="395" rx="9" fill="#f7fafc"/>';
    if(!circle){for(let x=35;x<490;x+=45)s+=line(x,18,x,377,'#edf1f5');for(let y=35;y<377;y+=45)s+=line(20,y,490,y,'#edf1f5');}
    edges.forEach(([a,b])=>{const active=focus===null||a===focus||b===focus;s+=path([coords[a],coords[b]],active?'#087f8c':'#c2ccd6',`opacity="${active?1:.25}" data-edge="${a}-${b}"`);});
    coords.forEach((p,i)=>{s+=`<circle cx="${p[0]}" cy="${p[1]}" r="8" fill="#f7fafc"/><circle cx="${p[0]}" cy="${p[1]}" r="${i===focus?6:4}" fill="${i===focus?'#efb752':'#fff'}" stroke="#182b45" stroke-width="1.2"><title>${safe(D.names[i])} · degree ${degree[i]}</title></circle>`+txt(p[0]+8,p[1]-7,i,'font-weight="600"');});
    if(circle)s+=txt(255,193,'VISITING ORDER','text-anchor="middle" font-weight="700"')+txt(255,214,'One city · two neighbors','text-anchor="middle"');
    $('map').innerHTML=s;$('mapLabel').textContent=`${title} · cost ${fmt(cost)}`;
  }
  function drawCurve(f){
    let s='';
    for(let k=0;k<4;k++){let v=vmin+(vmax-vmin)*k/3,y=cy(v);s+=line(52,y,490,y)+txt(44,y+4,fmt(v),'text-anchor="end"');}
    for(let k=0;k<3;k++){let e=1+(D.total-1)*k/2;s+=txt(cx(e),173,Math.round(e),'text-anchor="middle"');}
    s+=txt(270,193,'Objective evaluations','text-anchor="middle"');
    const events=D.events.filter(e=>e.evaluations<=f.evaluations);
    s+=path(events.map(e=>[cx(e.evaluations),cy(e.cost)]),'#087f8c');
    s+=path(events.map(e=>[cx(e.evaluations),cy(e.best)]),'#7059b8','stroke-dasharray="5 4"');
    s+=line(cx(f.evaluations),15,cx(f.evaluations),154,'#b1becc','stroke-dasharray="3 4"');
    s+=`<circle cx="${cx(f.evaluations)}" cy="${cy(f.cost)}" r="4" fill="#087f8c"/>`;
    $('curve').innerHTML=s;
  }
  function drawDiagnostic(f){
    let s='';
    if(ga&&f.population_costs){
      const costs=f.population_costs,lo=Math.min(...costs),hi=Math.max(...costs),bins=12,counts=Array(bins).fill(0);
      costs.forEach(v=>counts[Math.min(bins-1,Math.floor((v-lo)/Math.max(1,hi-lo)*bins))]++);
      const max=Math.max(...counts);
      counts.forEach((c,i)=>{const h=c/max*49;s+=`<rect x="${35+i*35}" y="${69-h}" width="29" height="${h}" fill="#7059b8" opacity=".7"/>`;});
      s+=txt(35,12,'Population cost distribution · count')+txt(35,86,fmt(lo))+txt(451,86,fmt(hi),'text-anchor="end"');
      s+=txt(35,108,`Distinct cycles: ${fmt(f.diversity*100)}% (rotations / reversals identified)`);
    }else if(ga){
      s+=txt(15,24,'Record population and cached costs to show GA diagnostics.');
    }else if(sa&&f.temperature!==null){
      const temp=f.temperature||0;
      s+=txt(15,19,`Temperature: ${fmt(temp)}`)+txt(495,19,rawMaxT>0?`T / maximum T = ${fmt(temp/maxT*100)}%`:'No positive temperature recorded','text-anchor="end"');
      s+='<rect x="15" y="30" width="480" height="10" rx="5" fill="#e1e8ef"/>';
      s+=`<rect x="15" y="30" width="${480*temp/maxT}" height="10" rx="5" fill="#d26532"/>`;
      const past=D.events.filter(e=>e.evaluations<=f.evaluations&&e.delta>0);
      s+=txt(15,66,`Worse proposals accepted so far: ${past.filter(e=>e.accepted).length} / ${past.length}`);
      s+=txt(15,91,'Explain the decision using your acceptance rule and schedule.');
    }else if(sa){
      const past=D.events.filter(e=>e.evaluations<=f.evaluations&&e.delta>0);
      s+=txt(15,24,`Worse proposals accepted so far: ${past.filter(e=>e.accepted).length} / ${past.length}`);
      s+=txt(15,52,'Every rise of the current curve is an accepted worse move.');
      s+=txt(15,80,'Explain the decision using your acceptance rule and schedule.');
    }else if(demo){
      const past=D.events.filter(e=>e.evaluations<=f.evaluations&&e.accepted!==null);
      s+=txt(15,24,`Accepted proposals so far: ${past.filter(e=>e.accepted).length} / ${past.length}`);
      s+=txt(15,52,'Tour menu: Candidate shows the proposal, Best archived the archive.');
      s+=txt(15,80,'Your algorithms will supply their own tours and diagnostics.');
    }else{
      const past=D.events.filter(e=>e.evaluations<=f.evaluations&&e.accepted!==null);
      s+=txt(15,24,`Accepted recorded proposals: ${past.filter(e=>e.accepted).length} / ${past.length}`);
      s+=txt(15,52,'A rejected proposal alone does not certify a local optimum.');
      s+=txt(15,80,'State your neighborhood, selection rule, and stopping condition.');
    }
    $('diagnostic').innerHTML=s;
  }
  function render(){
    const f=F[index];$('cost').textContent=fmt(f.cost);$('best').textContent=fmt(f.best);$('evaluation').textContent=`${f.evaluations} / ${D.total}`;
    $('scrub').value=index;$('frame').textContent=`Frame ${index+1} / ${F.length}${ga&&f.generation!==null?' · generation '+f.generation:''}${index===F.length-1?` · end of the run · ${D.steps} steps`:''}`;
    let title=f.accepted===null?'Recorded state':f.accepted?'Accepted':'Rejected';
    const numbers=f.delta===null?'':`Δ = ${fmt(f.delta)} · candidate cost ${fmt(f.candidate_cost)}`;
    let detail=f.note||numbers||'Recorded state.';
    if(sa&&f.probability!==null)detail+=` · acceptance probability ${f.probability.toFixed(3)}`;
    if(ga){title=f.generation===null?'Population state':`Generation ${f.generation}`;detail=f.note||'Recorded after your population update.';}
    if(demo){title+=' · guided example';detail=numbers?`${numbers} · ${f.note}`:f.note;}
    $('decision').innerHTML=`<b>${safe(title)}</b>${safe(detail)}`;
    drawMap(f);drawCurve(f);drawDiagnostic(f);
  }
  function stop(){if(timer)clearInterval(timer);timer=null;$('play').textContent='Play';$('play').setAttribute('aria-label','Play animation');}
  function play(){if(index===F.length-1)index=0;$('play').textContent='Pause';$('play').setAttribute('aria-label','Pause animation');render();timer=setInterval(()=>{if(index>=F.length-1){stop();return;}index++;render();},Number($('speed').value));}
  $('play').onclick=()=>timer?stop():play();
  $('back').onclick=()=>{stop();index=Math.max(0,index-1);render();};
  $('forward').onclick=()=>{stop();index=Math.min(F.length-1,index+1);render();};
  $('scrub').oninput=()=>{stop();index=Number($('scrub').value);render();};
  $('speed').onchange=()=>{if(timer){stop();play();}};
  for(const id of ['layout','route','focus'])$(id).onchange=()=>{stop();render();};
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  render();
})();
