import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';
const source=readFileSync(process.argv[2]||new URL('../../.venv-xr/lib/python3.10/site-packages/vuer/client_build/assets/xr-session-fix/chunks/chunk-Dd3xtWba.js',import.meta.url),'utf8');
function span(start,end){const a=source.indexOf(start);assert(a>=0,start);const b=source.indexOf(end,a);assert(b>=0,end);return source.slice(a,b)}
const button=span('XRButton=({store:K})=>',',iframeVuerLinkButton=');
const enter=span('async function enterXRSession','async function setupXRSession');
const offer=span('async function offerSession','async function setFrameRate');
const setup=span('async function setupXRSession','const allSessionModes=');
const build=span('function buildXRSessionInit','function addXRSessionFeature');
const setSession=span('this.setSession=async function(lt)',',this.getEnvironmentBlendMode=');
let passed=0;
async function test(name,fn){await fn();console.log('PASS',name);passed++}
function fixture({supported=true,blend='alpha-blend',setupError=false}={}){
 const events=[];
 const session={environmentBlendMode:blend,end:async()=>events.push('end')};
 const manager={setSession:async s=>{assert.equal(s,session);events.push('setSession');if(setupError)throw Error('reference space rejected')}};
 const context={navigator:{xr:{isSessionSupported:async()=>supported,requestSession:async()=>{events.push('requestSession');return session},offerSession:async()=>{events.push('offerSession');return session}}},buildXRSessionInit:()=>({}),setupXRSession:async (s,m)=>m.setSession(s),showTeleVuerXRStatus:s=>events.push(['status',s]),showTeleVuerXRError:(m,e)=>events.push(['error',m,e.message]),window:{},console};
 vm.createContext(context);vm.runInContext(enter+offer,context);
 return {context,events,manager,session};
}
await test('unsupported AR is disabled by preflight before a user click',async()=>{
 let state=null;const errors=[];
 const context={URLSearchParams,location:{search:'?xrMode=immersive-ar'},window:{},navigator:{xr:{isSessionSupported:async()=>false}},reactExports:{useState:()=>[state,s=>state=s],useEffect:cb=>cb()},jsxRuntimeExports:{jsx:(type,props)=>({type,props})},style$5:'',buttonStyle$2:'',showTeleVuerXRError:(m,e)=>errors.push(e)};
 vm.createContext(context);vm.runInContext('var '+button+';',context);
 assert.equal(context.XRButton({store:{}}).props.children.props.disabled,true);
 await Promise.resolve();
 assert.equal(context.XRButton({store:{}}).props.children.props.disabled,true);
 assert.equal(context.XRButton({store:{}}).props.children.props.children,'WebXR unavailable');
 assert(errors.length>0);
});
await test('requestSession starts synchronously within the click activation',async()=>{const f=fixture();const pending=f.context.enterXRSession(null,'immersive-ar',{},f.manager);assert(f.events.includes('requestSession'));await pending});
await test('opaque AR session closes and reports unavailable passthrough',async()=>{const f=fixture({blend:'opaque'});await assert.rejects(f.context.enterXRSession(null,'immersive-ar',{},f.manager),/opaque AR/);assert(f.events.includes('end'));assert(!f.events.includes('setSession'))});
await test('setup failure ends session before showing the error',async()=>{const f=fixture({setupError:true});await assert.rejects(f.context.enterXRSession(null,'immersive-ar',{},f.manager),/reference space/);assert(f.events.indexOf('end')<f.events.findIndex(x=>x[0]==='error'))});
await test('AR enters through WebXRManager without reading a renderer context',async()=>{const f=fixture();assert.equal(await f.context.enterXRSession(null,'immersive-ar',{},f.manager),f.session);assert.equal(f.events.filter(x=>x==='setSession').length,1);assert(!f.events.includes('end'))});
await test('VR accepts opaque sessions',async()=>{const f=fixture({blend:'opaque'});assert.equal(await f.context.enterXRSession(null,'immersive-vr',{},f.manager),f.session)});
await test('offered sessions await setup and clean up failed initialization',async()=>{const f=fixture({setupError:true});await f.context.offerSession(f.manager,{offerSession:'immersive-ar'},null);assert(f.events.includes('end'));assert(f.events.some(x=>x[0]==='error'))});
await test('AR requires its actual floor reference and preserves status overlay',async()=>{const status={},overlay={appendChild:x=>assert.equal(x,status)};const context={document:{getElementById:id=>id==='televuer-xr-overlay'?overlay:status}};vm.createContext(context);vm.runInContext(build,context);const init=context.buildXRSessionInit('immersive-ar',null,{});assert.deepEqual(Array.from(init.requiredFeatures),['local-floor']);assert(init.optionalFeatures.includes('hand-tracking'));assert(!init.optionalFeatures.includes('layers'));assert.equal(init.domOverlay.root,overlay)});
await test('Three owns one AR layer even when projection API exists without layers feature',async()=>{
 const events=[];let submitted;
 const gl={makeXRCompatible:async()=>{},getContextAttributes:()=>({alpha:true,xrCompatible:true})};
 const renderer={getRenderTarget:()=>null,getPixelRatio:()=>1,getSize:()=>{},setPixelRatio:()=>{},setSize:()=>{}};
 const session={environmentBlendMode:'alpha-blend',addEventListener:()=>{},updateRenderState:s=>{submitted=s.baseLayer;events.push(s)},requestReferenceSpace:async t=>{assert.equal(t,'local-floor');return {}}};
 class Layer{constructor(s,g,init){assert.equal(s,session);assert.equal(g,gl);assert.equal(init.alpha,true);this.framebufferWidth=10;this.framebufferHeight=10;this.framebuffer={}}}
 class Binding{createProjectionLayer(){throw Error('layers feature was not enabled')}}
 const context={session,renderer,gl,XRWebGLLayer:Layer,XRWebGLBinding:Binding,WebGLRenderTarget:class{},RGBAFormat:1,UnsignedByteType:2};
 vm.createContext(context);
 vm.runInContext(`var ne=null,Ae=null,be=null,Te={},me={xrCompatible:true,antialias:true,depth:false,stencil:false},pe=null,de=null,ye=null,le=null,oe=null,se="local-floor",ie=1,ae=1,fe=null;var _=renderer,Z=gl;function ke(){}function Ge(){}function He(){}var ht={setContext(){},start(){}};var manager={setFoveation(){},dispatchEvent(){}};var te=manager;(function(){${setSession}}).call(manager);`,context);
 await context.manager.setSession(session);
 assert.equal(events.length,1);assert(submitted instanceof Layer);assert.equal(context.pe,submitted);assert.equal(context.de,null);assert.equal(context.manager.isPresenting,true);
});
console.log(`${passed} behavioral checks passed`);
