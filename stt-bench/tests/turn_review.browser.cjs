// Synthetic-only listening editor test. No real review is marked approved.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const http=require('node:http');
(async()=>{
 const root=path.resolve(process.argv[2]);
 const server=http.createServer((req,res)=>{
  const p=path.resolve(root,'.'+decodeURIComponent(req.url.split('?')[0]==='/'?'/index.html':req.url.split('?')[0]));
  if(!p.startsWith(root+path.sep)){res.writeHead(403);res.end();return;}
  fs.readFile(p,(err,data)=>{if(err){res.writeHead(404);res.end();return;}res.setHeader('Content-Type',p.endsWith('.html')?'text/html':p.endsWith('.wav')?'audio/wav':'application/json');res.end(data);});
 });
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 let browser;
 try{
  browser=await chromium.launch({headless:true,channel:process.env.PLAYWRIGHT_CHANNEL||'chrome'});
  const context=await browser.newContext({viewport:{width:1360,height:1000}}),page=await context.newPage(),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await context.route('**/*',route=>new URL(route.request().url()).hostname==='127.0.0.1'?route.continue():route.abort());
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const data=JSON.parse(await page.locator('#turn-data').textContent());assert.equal(data.draft.sources.length,2);
  const count=await page.locator('#turns option').count();assert.equal(count,2);
  await page.locator('#context').click();await page.waitForFunction(()=>document.querySelector('#clock').textContent.includes('Conversation'));
  await page.locator('#stop').click();
  await page.locator('#play').click();await page.waitForFunction(()=>document.querySelector('#clock').textContent.includes('Conversation'));
  await page.locator('#stop').click();
  await page.locator('#split').click();assert.equal(await page.locator('#turns option').count(),count+1);
  const first=data.review.turns[0].turn_id;await page.locator('#turns').selectOption(first);
  await page.locator('#merge').click();assert.equal(await page.locator('#turns option').count(),count);
  await page.locator('#reviewer').fill('Synthetic browser test');
  await page.locator('#boundary').check();await page.locator('#transcript').check();
  await page.locator('#reference').fill('yes thank you');await page.locator('#reference').blur();
  assert.equal(await page.locator('#transcript').isChecked(),false);
  await page.locator('#reference').fill('yes please');await page.locator('#reference').blur();
  await page.locator('#notes').fill('Synthetic split and merge verified');await page.locator('#notes').blur();
  await page.locator('#boundary').check();await page.locator('#transcript').check();
  const [download]=await Promise.all([page.waitForEvent('download'),page.locator('#export').click()]);
  const saved=path.join(root,'browser-review.json');await download.saveAs(saved);
  const review=JSON.parse(fs.readFileSync(saved));const units=review.turns.flatMap(t=>t.unit_ids);
  assert.equal(units.length,Object.keys(data.draft.units).length);assert.equal(new Set(units).size,units.length);
  assert.equal(review.turns.length,2);assert.equal(review.draft_sha256,data.review.draft_sha256);
  await page.locator('#import').setInputFiles(saved);await page.waitForFunction(()=>document.querySelector('#status').textContent==='Saved review loaded.');
  assert.equal(await page.locator('#reviewer').inputValue(),'Synthetic browser test');
  await page.setViewportSize({width:390,height:844});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  assert.deepEqual(errors,[]);
  console.log('PASS: aligned and isolated playback, split, merge, approval invalidation, export/import, word coverage, mobile layout, and zero remote requests.');
 }finally{if(browser)await browser.close();await new Promise(r=>server.close(r));}
})().catch(e=>{console.error(e);process.exitCode=1});
