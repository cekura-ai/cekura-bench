// Offline browser checks for the simplified single-page report.
// NODE_PATH may point to an existing Playwright installation.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
(async()=>{
  const html=path.resolve(process.argv[2]||'reports/benchmark-dashboard/index.html');
  const output=path.dirname(html),browser=await chromium.launch({headless:true});
  try {
    const context=await browser.newContext({viewport:{width:1440,height:1100},offline:true,acceptDownloads:true});
    const page=await context.newPage(),errors=[],remote=[];
    context.on('page',p=>p.on('pageerror',e=>errors.push(e.message)));
    page.on('pageerror',e=>errors.push(e.message));
    context.on('request',r=>{if(/^https?:/.test(r.url()))remote.push(r.url());});
    await page.goto(pathToFileURL(html).href);
    const data=JSON.parse(await page.locator('#benchmark-data').textContent());
    assert.equal(await page.locator('#table-body tr').count(),10);
    assert.equal(await page.locator('#public-benchmark .chart svg').count(),2);
    assert.equal(await page.locator('#scatter-chart circle').count(),10);
    assert.equal(await page.locator('#about').getAttribute('open'),null);
    assert.equal(await page.locator('select,input').count(),0);
    assert.equal(await page.locator('#best-wer').textContent(),'1.59%');
    assert.equal(await page.locator('#best-latency').textContent(),'80 ms');
    for(const m of data.models){
      const row=page.locator('#table-body tr').filter({has:page.getByText(m.label,{exact:true})});
      assert.equal(await row.locator('td').nth(1).textContent(),(m.final.wer*100).toFixed(2)+'%');
      assert.equal(await row.locator('td').nth(4).textContent(),m.usable.toLocaleString('en-US')+' / 1,000');
    }
    await page.screenshot({path:path.join(output,'simple-desktop.png'),fullPage:true});
    await page.screenshot({path:path.join(output,'simple-desktop-top.png')});
    await page.locator('#scatter-chart').screenshot({path:path.join(output,'simple-scatter.png')});
    await page.locator('#metric-latency').click();
    assert.equal(await page.locator('#metric-latency').getAttribute('aria-pressed'),'true');
    assert.match(await page.locator('#comparison-subtitle').textContent(),/Median finalize/);
    await page.locator('#comparison-chart [data-tip]').first().focus();
    assert.equal(await page.locator('#tooltip').isVisible(),true);
    assert.match(await page.locator('#tooltip').textContent(),/measured clips/);
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('#tooltip').isVisible(),false);
    await page.locator('#scatter-chart circle').first().scrollIntoViewIfNeeded();
    await page.mouse.move(0,0);
    await page.locator('#scatter-chart circle').first().hover();
    assert.equal(await page.locator('#tooltip').isVisible(),true);
    await page.locator('#legend button').first().click();
    assert.equal(await page.locator('#table-body tr').count(),9);
    assert.equal(await page.locator('#scatter-chart circle').count(),9);
    assert.equal(await page.locator('#best-wer').textContent(),'1.59%');
    await page.locator('[data-sort="wer"]').click();
    assert.equal(await page.locator('th[data-key="wer"]').getAttribute('aria-sort'),'descending');
    assert.match(await page.locator('#table-body tr').first().textContent(),/Flux Multilingual/);
    const waiting=page.waitForEvent('download');await page.locator('#export').click();
    const download=await waiting,csv=await fs.readFile(await download.path(),'utf8');
    assert.equal(csv.split('\r\n').length,10);
    assert.ok(csv.includes('Unavailable')&&csv.includes('Completion p50 ms'));
    const activeIds=await page.locator('#legend button[aria-pressed=true]').evaluateAll(nodes=>nodes.map(n=>n.dataset.model));
    for(const id of activeIds)await page.locator(`#legend button[data-model="${id}"]`).click();
    assert.equal(await page.locator('#empty').isVisible(),true);
    assert.equal(await page.locator('#export').isDisabled(),true);
    await page.locator('#restore').click();
    assert.equal(await page.locator('#table-body tr').count(),10);
    await page.locator('#about summary').click();
    assert.match(await page.locator('#sources').textContent(),/unformatted/);
    const mobile=await context.newPage();await mobile.setViewportSize({width:390,height:844});
    await mobile.goto(pathToFileURL(html).href);
    assert.equal(await mobile.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    await mobile.screenshot({path:path.join(output,'simple-mobile.png'),fullPage:true});
    await mobile.screenshot({path:path.join(output,'simple-mobile-top.png')});
    await mobile.locator('#metric-latency').click();
    await mobile.locator('#legend button').first().click();
    assert.equal(await mobile.locator('#table-body tr').count(),9);
    await mobile.locator('#legend button').first().focus();await mobile.keyboard.press('Enter');
    assert.equal(await mobile.locator('#table-body tr').count(),10);
    assert.deepEqual(errors,[]);assert.deepEqual(remote,[]);
    console.log('PASS: offline desktop/mobile, source totals, highlights, metric switch, scatter, legend filtering, sorting, tooltips, empty state, CSV and methodology.');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
