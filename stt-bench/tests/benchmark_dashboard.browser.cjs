// Offline checks against the actual generated unified dashboard.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
(async()=>{
 const file=path.resolve(process.argv[2]||'reports/benchmark-dashboard/index.html');
 const browser=await chromium.launch({headless:true,channel:process.env.PLAYWRIGHT_CHANNEL||undefined});
 try {
  const context=await browser.newContext({viewport:{width:1440,height:1100},offline:true,acceptDownloads:true});
  const page=await context.newPage(),errors=[],remote=[];
  context.on('request',r=>{if(/^https?:/.test(r.url()))remote.push(r.url());});
  context.on('page',p=>p.on('pageerror',e=>errors.push(e.message)));
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(file).href);
  assert.deepEqual(errors,[]);
  const data=JSON.parse(await page.locator('#benchmark-data').textContent());
  assert.equal(data.models.length,16);
  assert.equal(data.models.filter(m=>m.rankable).length,13);
  const assembly=data.models.find(m=>m.id==='assemblyai-universal-3-5-pro-min-latency');
  assert.ok(assembly.terminal&&assembly.rankable&&assembly.rank>0);
  assert.equal(assembly.cohorts.pipecat.usable,1000);
  assert.equal(assembly.cohorts.private.usable,8);
  assert.equal(assembly.combined.wer,991/36412);
  assert.equal(assembly.reliability.failed,0);
  assert.equal(assembly.cohorts.fleurs.attempted,0);
  assert.match(await page.locator('#status').textContent(),/full run verified complete/);
  assert.equal(await page.locator('#table-body tr').count(),16);
  assert.equal(await page.locator('#benchmark-switch').count(),0);
  assert.equal(await page.locator('[data-sort="cost"]').textContent(),'Actual cost (USD)');
  assert.ok(data.models.every(m=>m.actual_cost_usd===null&&m.actual_cost_status==='unverified'));
  assert.equal(await page.locator('#table-body tr td:nth-child(4)').allTextContents().then(c=>c.every(v=>v==='Unverified')),true);
  assert.ok(!data.models.some(m=>data.removed_models.includes(m.id)));
  assert.equal(await page.locator('[data-sort="failure"]').textContent(),'Failure rate');
  assert.equal(await page.locator('[data-sort="notrun"]').textContent(),'Not run');
  for(const m of data.models){
   const r=m.reliability;
   assert.equal(r.planned,1188);
   assert.equal(r.failed+r.not_run+r.usable,1188);
   assert.equal(r.failure_rate,r.failed/1188);
   const cells=page.locator(`#table-body tr[data-model="${m.id}"] td`);
   assert.equal(await cells.nth(4).textContent(),(r.failure_rate*100).toFixed(2)+'%');
   assert.equal(await cells.nth(5).textContent(),(r.not_run_rate*100).toFixed(2)+'%');
  }
  await page.locator('#metric-failure').click();
  assert.match(await page.locator('#comparison-subtitle').textContent(),/1,188 planned/);
  assert.equal(await page.locator('#metric-failure').getAttribute('aria-pressed'),'true');
  await page.locator('[data-sort="failure"]').click();
  const failureOrder=await page.locator('#table-body tr').evaluateAll(rs=>rs.map(r=>r.dataset.model));
  const rates=failureOrder.map(id=>data.models.find(m=>m.id===id).reliability.failure_rate);
  assert.deepEqual(rates,[...rates].sort((a,b)=>a-b));
  const nums=v=>v==null?'Unavailable':Number(v).toLocaleString('en-US',{maximumFractionDigits:0});
  for(const metric of ['final','interim','word','completion']){
   await page.locator('#timing').selectOption(metric);
   for(const m of data.models){
    const cells=page.locator(`#table-body tr[data-model="${m.id}"] td`);
    assert.equal(await cells.nth(2).textContent(),m.combined.wer==null?'Unavailable':(m.combined.wer*100).toFixed(2)+'%');
    for(const [i,p] of [50,90,95,99].entries())assert.equal(await cells.nth(6+i).textContent(),nums(m.timings[metric]['p'+p+'_ms']));
   }
  }
  await page.locator('#timing').selectOption('interim');
  await page.locator('#metric-latency').click();
  await page.locator('#percentile').selectOption('99');
  assert.match(await page.locator('#comparison-subtitle').textContent(),/Interim.*p99/);
  assert.match(await page.locator('#timing-help').textContent(),/Earlier partials are not measured/);
  const bar=page.locator('#comparison-chart [data-tip]').first();await bar.focus();
  assert.equal(await page.locator('#tooltip').isVisible(),true);
  await page.keyboard.press('Escape');assert.equal(await page.locator('#tooltip').isVisible(),false);
  await page.locator('#legend button').first().click();
  assert.equal(await page.locator('#table-body tr').count(),15);
  await page.locator('[data-sort="wer"]').click();await page.locator('[data-sort="wer"]').click();
  assert.equal(await page.locator('th[data-key="wer"]').getAttribute('aria-sort'),'descending');
  const waiting=page.waitForEvent('download');await page.locator('#export').click();
  const download=await waiting,csv=await fs.readFile(await download.path(),'utf8');
  assert.equal(csv.split('\r\n').length,16);
  assert.ok(csv.includes('Interim update after speech end p99 ms'));
  assert.ok(csv.includes('Actual cost USD')&&csv.includes('Actual cost status')&&csv.includes('unverified'));
  assert.ok(csv.includes('Failure rate (failed / planned)')&&csv.includes('Not run rate'));
  assert.ok(csv.includes('Source SHA-256')&&csv.includes('Unavailable'));
  const ids=await page.locator('#legend button[aria-pressed=true]').evaluateAll(bs=>bs.map(b=>b.dataset.model));
  for(const id of ids)await page.locator(`#legend button[data-model="${id}"]`).click();
  assert.equal(await page.locator('#empty').isVisible(),true);assert.equal(await page.locator('#export').isDisabled(),true);
  await page.locator('#restore').click();assert.equal(await page.locator('#table-body tr').count(),16);
  await page.locator('#metric-wer').click();await page.locator('#timing').selectOption('final');await page.locator('#percentile').selectOption('50');
  await page.locator('[data-sort="rank"]').click();
  await page.screenshot({path:path.join(path.dirname(file),'unified-desktop.png'),fullPage:true});
  const mobile=await context.newPage();await mobile.setViewportSize({width:390,height:844});
  await mobile.goto(pathToFileURL(file).href);
  assert.equal(await mobile.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  await mobile.locator('#timing').selectOption('interim');
  await mobile.locator('#legend button').first().focus();await mobile.keyboard.press('Enter');
  assert.equal(await mobile.locator('#table-body tr').count(),15);
  await mobile.screenshot({path:path.join(path.dirname(file),'unified-mobile.png'),fullPage:true});
  assert.deepEqual(errors,[]);assert.deepEqual(remote,[]);
  console.log('PASS: all 4 timing metrics and percentiles, 16 rows, exclusions, sorting, filtering, tooltips, empty state, CSV, offline desktop/mobile.');
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
