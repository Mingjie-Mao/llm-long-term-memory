"""Self-contained local UI for a hash-bound human review packet."""

# Chinese UI copy and embedded HTML/JS retain their own line layout.
# ruff: noqa: E501, RUF001

from __future__ import annotations

import json


def render(packet: dict) -> str:
    payload = json.dumps(packet, ensure_ascii=False).replace("<", "\\u003c")
    return HTML.replace("__PACKET__", payload)


HTML = r"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>计数证据审核</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f6f8;color:#243247;font:16px/1.6 system-ui,sans-serif}
header,main{max-width:1120px;margin:auto;padding:24px}header{position:sticky;top:0;background:#f5f6f8ed;z-index:2;border-bottom:1px solid #d6dce5}
h1{font-size:26px;margin:0 0 6px}h2{font-size:19px}p{margin:8px 0}.muted{color:#64748b;font-size:14px}
button,select,input,textarea{font:inherit;border:1px solid #bac5d3;border-radius:6px;padding:8px;background:white;color:inherit}
button{cursor:pointer;background:#145c54;color:white;border:0}button.secondary{background:#e3e9ef;color:#243247}
textarea{display:block;width:100%;min-height:75px;margin:8px 0}input[type=text]{width:240px}label{display:inline-block;margin:5px 12px 5px 0}
details{background:white;border:1px solid #dce1e9;border-radius:9px;padding:15px;margin:13px 0}summary{cursor:pointer;font-weight:600}
.source{background:#f4f7fa;border-left:3px solid #7d9eaa;padding:12px;margin:10px 0;white-space:pre-wrap;overflow-wrap:anywhere}
.flag{color:#9b361c;background:#fff2e8;padding:10px;border-radius:5px}.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.member{display:grid;grid-template-columns:1fr 1fr;gap:8px;background:#f4f7fa;padding:12px;margin:8px 0}.member input{width:100%}.member textarea{grid-column:1/-1}
#status{font-weight:600}code{font-size:13px;overflow-wrap:anywhere}li{margin:8px 0}
</style><header><h1>计数证据审核</h1><p class="muted">只审核问题、成员与原文；不展示候选回答或分数。内容仅在本机保存。</p>
<div class="row"><span id="status"></span><button id="export">导出裁定 JSON</button><button class="secondary" id="import">导入已保存裁定</button><input id="file" type="file" accept=".json" hidden></div>
</header><main><p id="summary"></p><div class="row"><label>审阅者 <input id="reviewer" type="text" placeholder="填写姓名或稳定标识"></label><label><input type="checkbox" id="policy"> 我已审核并同意下面的规则与关系允许表</label></div>
<details><summary>规则与关系允许表（先审核）</summary><div id="rules"></div></details>
<p>逐题选择“接受 / 排除 / 重写”，填写理由。接受实体题时，逐个成员填写名称、来源编号和原文引用，并检查提供的全部原文是否还遗漏成员。红色标记题需排除或重写；重写题将进入下一版，不直接改变本次金标。</p>
<p class="muted">旧裁定的“接受”表示接受该成员属于旧问句集合，不表示认可旧的记忆行计数单位。来源池是围绕记忆锚点的原文及相邻轮次，不能代表用户一生全部行为。页面自动保存到当前浏览器；请另导出文件保存。</p>
<div id="items"></div></main><script type="application/json" id="packet">__PACKET__</script><script>
'use strict';
const p=JSON.parse(document.getElementById('packet').textContent),key='count-review:'+p.packet_sha256;
let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
let decisions=new Map((saved.decisions||[]).map(x=>[x.id,x]));
const $=id=>document.getElementById(id),el=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n};
$('reviewer').value=saved.reviewer||'';$('policy').checked=saved.policy_approved===true;
const ul=el('ol');p.policy.rules.forEach(r=>ul.append(el('li',r)));$('rules').append(ul);
Object.entries(p.policy.relations).forEach(([k,v])=>{$('rules').append(el('h2',k+' · '+v.unit));$('rules').append(el('p',v.predicates.join(', ')))});
$('summary').textContent=`可审核：${p.summary.reviewable.entity||0} 道实体题、${p.summary.reviewable.legacy||0} 条旧裁定。${p.quarantine.length} 项因封存边界等原因隔离，不展示原文。`;
function record(){return {schema_version:1,packet_sha256:p.packet_sha256,reviewer_kind:'human',reviewer:$('reviewer').value.trim(),policy_approved:$('policy').checked,reviewed_at:new Date().toISOString(),decisions:p.items.map(i=>decisions.get(i.id)||{id:i.id,verdict:'',reason:'',members:[],scope_complete:false})}}
function save(){const r=record();try{localStorage.setItem(key,JSON.stringify(r))}catch(e){$('status').textContent='本地存储失败，请导出文件';return}const done=r.decisions.filter(x=>x.verdict&&x.reason.trim()).length;$('status').textContent=`已填写 ${done} / ${p.items.length}`}
function draw(){ const opened=new Set([...document.querySelectorAll('#items>details[open]')].map(x=>x.dataset.id));$('items').replaceChildren(); p.items.forEach((i,index)=>{
 let d=decisions.get(i.id);if(!d){d={id:i.id,verdict:'',reason:'',members:[],scope_complete:false};decisions.set(i.id,d)}
 const card=el('details'),title=el('summary',`${index+1}. ${i.kind==='entity'?'实体题':'旧成员'} · ${i.id}`);card.dataset.id=i.id;card.open=opened.has(i.id);card.append(title,el('p',i.question));
 if(i.flags.length){const f=el('p',i.flags.join('；'));f.className='flag';card.append(f)}
 const ms=el('details');ms.append(el('summary','提取记忆'));i.memories.forEach(m=>ms.append(el('p',`${m.predicate} / ${m.scope} / ${m.status}: ${m.content}`)));card.append(ms);
 const ss=el('details');ss.append(el('summary',`原文证据（${i.sources.length} 条）`));i.sources.forEach(s=>{const n=el('div',`${s.id} · ${s.role} · ${s.recorded_at||'日期未定'}\n${s.text}`);n.className='source';ss.append(n)});card.append(ss);
 // Anything resembling a conclusion is rendered only AFTER the evidence. A prior
 // verdict exists on legacy items alone, where re-reviewing that model decision is
 // the whole item; entity questions carry none, so no model answer precedes a gold
 // decision. Rule 0.
 if(i.proposed_entities.length)card.append(el('p','旧生成器的成员（待逐个核对，不是答案）：'+i.proposed_entities.join('；')));
 if(i.prior_model_verdict){const v=el('p','待复核的模型裁定（是被审对象，不是结论）：'+i.prior_model_verdict);v.className='flag';card.append(v)}
 const select=el('select');[['','请选择裁定'],['accept','接受'],['exclude','排除'],['rewrite','问题或单位需重写']].forEach(([v,t])=>{const o=el('option',t);o.value=v;select.append(o)});select.value=d.verdict;select.onchange=()=>{d.verdict=select.value;save()};card.append(select);
 const reason=el('textarea');reason.placeholder='裁定理由（必填）';reason.value=d.reason||'';reason.oninput=()=>{d.reason=reason.value;save()};card.append(reason);
 if(i.kind==='entity'){
  const label=el('label'),check=el('input');check.type='checkbox';check.checked=d.scope_complete===true;check.onchange=()=>{d.scope_complete=check.checked;save()};label.append(check,document.createTextNode(' 已核对全部提供原文，确认集合完整'));card.append(label);
  const list=el('div');card.append(list);
  function member(m,pos){const box=el('div');box.className='member';const name=el('input');name.value=m.entity||'';name.placeholder='成员名称';name.oninput=()=>{m.entity=name.value;save()};const source=el('select');const empty=el('option','选择来源编号');empty.value='';source.append(empty);i.sources.forEach(s=>{const o=el('option',s.id+' · '+s.role);o.value=s.id;source.append(o)});source.value=m.source_id||'';source.onchange=()=>{m.source_id=source.value;save()};const quote=el('textarea');quote.placeholder='粘贴支持此成员的原文引用，必须逐字相同';quote.value=m.quote||'';quote.oninput=()=>{m.quote=quote.value;save()};const remove=el('button','删除此成员');remove.className='secondary';remove.onclick=()=>{d.members.splice(pos,1);save();draw()};box.append(name,source,quote,remove);list.append(box)}
  (d.members||[]).forEach(member);const add=el('button','添加一个成员');add.className='secondary';add.onclick=()=>{d.members.push({entity:'',source_id:'',quote:''});save();draw()};card.append(add);
  const zero=el('textarea');zero.placeholder='只有接受且为零时填写：来源编号 | 明确没有成员的原文引用';zero.value=d.zero_evidence?d.zero_evidence.source_id+' | '+d.zero_evidence.quote:'';zero.oninput=()=>{const at=zero.value.indexOf('|');d.zero_evidence=at<0?{}:{source_id:zero.value.slice(0,at).trim(),quote:zero.value.slice(at+1).trim()};save()};card.append(zero);
 }
 $('items').append(card);
 });save(); }
$('reviewer').oninput=save;$('policy').onchange=save;
$('export').onclick=()=>{save();const blob=new Blob([JSON.stringify(record(),null,2)+'\n'],{type:'application/json'}),a=el('a');a.href=URL.createObjectURL(blob);a.download='count-review-decisions.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
$('import').onclick=()=>$('file').click();$('file').onchange=async()=>{try{const r=JSON.parse(await $('file').files[0].text());if(r.packet_sha256!==p.packet_sha256)throw Error('文件不属于当前审核包');decisions=new Map(r.decisions.map(x=>[x.id,x]));$('reviewer').value=r.reviewer||'';$('policy').checked=r.policy_approved===true;draw()}catch(e){alert(e.message)}};
draw();
</script></html>"""
