(function(){var l;function aa(a,b){function c(){}
c.prototype=b.prototype;a.A=b.prototype;a.prototype=new c;a.prototype.constructor=a;for(var d in b)if(Object.defineProperties){var e=Object.getOwnPropertyDescriptor(b,d);e&&Object.defineProperty(a,d,e)}else a[d]=b[d]}
var ba="function"==typeof Object.defineProperties?Object.defineProperty:function(a,b,c){a!=Array.prototype&&a!=Object.prototype&&(a[b]=c.value)},ca="undefined"!=typeof window&&window===this?this:"undefined"!=typeof global&&null!=global?global:this;
function da(a,b){if(b){for(var c=ca,d=a.split("."),e=0;e<d.length-1;e++){var f=d[e];f in c||(c[f]={});c=c[f]}d=d[d.length-1];e=c[d];f=b(e);f!=e&&null!=f&&ba(c,d,{configurable:!0,writable:!0,value:f})}}
da("String.prototype.startsWith",function(a){return a?a:function(a,c){var b;if(null==this)throw new TypeError("The 'this' value for String.prototype.startsWith must not be null or undefined");if(a instanceof RegExp)throw new TypeError("First argument to String.prototype.startsWith must not be a regular expression");b=this+"";a+="";for(var e=b.length,f=a.length,g=Math.max(0,Math.min(c|0,b.length)),h=0;h<f&&g<e;)if(b[g++]!=a[h++])return!1;return h>=f}});
da("Reflect.apply",function(a){if(a)return a;var b=Function.prototype.apply;return function(a,d,e){return b.call(a,d,e)}});
da("Reflect.construct",function(a){return a?a:function(a,c,d){void 0===d&&(d=a);d=Object.create(d.prototype||Object.prototype);return Reflect.apply(a,d,c)||d}});
var n=this;function p(a){return void 0!==a}
function q(a,b,c){a=a.split(".");c=c||n;a[0]in c||!c.execScript||c.execScript("var "+a[0]);for(var d;a.length&&(d=a.shift());)!a.length&&p(b)?c[d]=b:c[d]&&c[d]!==Object.prototype[d]?c=c[d]:c=c[d]={}}
function r(a,b){for(var c=a.split("."),d=b||n,e;e=c.shift();)if(null!=d[e])d=d[e];else return null;return d}
function t(){}
function ea(a){a.ma=void 0;a.getInstance=function(){return a.ma?a.ma:a.ma=new a}}
function ha(a){var b=typeof a;if("object"==b)if(a){if(a instanceof Array)return"array";if(a instanceof Object)return b;var c=Object.prototype.toString.call(a);if("[object Window]"==c)return"object";if("[object Array]"==c||"number"==typeof a.length&&"undefined"!=typeof a.splice&&"undefined"!=typeof a.propertyIsEnumerable&&!a.propertyIsEnumerable("splice"))return"array";if("[object Function]"==c||"undefined"!=typeof a.call&&"undefined"!=typeof a.propertyIsEnumerable&&!a.propertyIsEnumerable("call"))return"function"}else return"null";
else if("function"==b&&"undefined"==typeof a.call)return"object";return b}
function ia(a){return"array"==ha(a)}
function ja(a){var b=ha(a);return"array"==b||"object"==b&&"number"==typeof a.length}
function u(a){return"string"==typeof a}
function ka(a){return"function"==ha(a)}
function la(a){var b=typeof a;return"object"==b&&null!=a||"function"==b}
var ma="closure_uid_"+(1E9*Math.random()>>>0),na=0;function oa(a,b,c){return a.call.apply(a.bind,arguments)}
function pa(a,b,c){if(!a)throw Error();if(2<arguments.length){var d=Array.prototype.slice.call(arguments,2);return function(){var c=Array.prototype.slice.call(arguments);Array.prototype.unshift.apply(c,d);return a.apply(b,c)}}return function(){return a.apply(b,arguments)}}
function v(a,b,c){Function.prototype.bind&&-1!=Function.prototype.bind.toString().indexOf("native code")?v=oa:v=pa;return v.apply(null,arguments)}
function w(a,b){var c=Array.prototype.slice.call(arguments,1);return function(){var b=c.slice();b.push.apply(b,arguments);return a.apply(this,b)}}
var x=Date.now||function(){return+new Date};
function z(a,b){function c(){}
c.prototype=b.prototype;a.A=b.prototype;a.prototype=new c;a.prototype.constructor=a;a.ob=function(a,c,f){for(var d=Array(arguments.length-2),e=2;e<arguments.length;e++)d[e-2]=arguments[e];return b.prototype[c].apply(a,d)}}
;var qa=document,A=window;function B(a){if(Error.captureStackTrace)Error.captureStackTrace(this,B);else{var b=Error().stack;b&&(this.stack=b)}a&&(this.message=String(a))}
z(B,Error);B.prototype.name="CustomError";var ra=String.prototype.trim?function(a){return a.trim()}:function(a){return a.replace(/^[\s\xa0]+|[\s\xa0]+$/g,"")};
function sa(a,b){return a<b?-1:a>b?1:0}
function ua(a){for(var b=0,c=0;c<a.length;++c)b=31*b+a.charCodeAt(c)>>>0;return b}
;var va=Array.prototype.indexOf?function(a,b,c){return Array.prototype.indexOf.call(a,b,c)}:function(a,b,c){c=null==c?0:0>c?Math.max(0,a.length+c):c;
if(u(a))return u(b)&&1==b.length?a.indexOf(b,c):-1;for(;c<a.length;c++)if(c in a&&a[c]===b)return c;return-1},C=Array.prototype.forEach?function(a,b,c){Array.prototype.forEach.call(a,b,c)}:function(a,b,c){for(var d=a.length,e=u(a)?a.split(""):a,f=0;f<d;f++)f in e&&b.call(c,e[f],f,a)},wa=Array.prototype.map?function(a,b,c){return Array.prototype.map.call(a,b,c)}:function(a,b,c){for(var d=a.length,e=Array(d),f=u(a)?a.split(""):a,g=0;g<d;g++)g in f&&(e[g]=b.call(c,f[g],g,a));
return e};
function xa(a,b){var c;a:{c=a.length;for(var d=u(a)?a.split(""):a,e=0;e<c;e++)if(e in d&&b.call(void 0,d[e],e,a)){c=e;break a}c=-1}return 0>c?null:u(a)?a.charAt(c):a[c]}
function ya(a,b){var c=va(a,b);0<=c&&Array.prototype.splice.call(a,c,1)}
function za(a){var b=a.length;if(0<b){for(var c=Array(b),d=0;d<b;d++)c[d]=a[d];return c}return[]}
function Aa(a,b){for(var c=1;c<arguments.length;c++){var d=arguments[c];if(ja(d)){var e=a.length||0,f=d.length||0;a.length=e+f;for(var g=0;g<f;g++)a[e+g]=d[g]}else a.push(d)}}
;function Ba(a,b){this.b=p(a)?a:0;this.f=p(b)?b:0}
Ba.prototype.equals=function(a){return a instanceof Ba&&(this==a?!0:this&&a?this.b==a.b&&this.f==a.f:!1)};
Ba.prototype.ceil=function(){this.b=Math.ceil(this.b);this.f=Math.ceil(this.f);return this};
Ba.prototype.floor=function(){this.b=Math.floor(this.b);this.f=Math.floor(this.f);return this};
Ba.prototype.round=function(){this.b=Math.round(this.b);this.f=Math.round(this.f);return this};function Ca(a,b){this.width=a;this.height=b}
l=Ca.prototype;l.aspectRatio=function(){return this.width/this.height};
l.isEmpty=function(){return!(this.width*this.height)};
l.ceil=function(){this.width=Math.ceil(this.width);this.height=Math.ceil(this.height);return this};
l.floor=function(){this.width=Math.floor(this.width);this.height=Math.floor(this.height);return this};
l.round=function(){this.width=Math.round(this.width);this.height=Math.round(this.height);return this};function Da(a){var b=Ea,c;for(c in b)if(a.call(void 0,b[c],c,b))return c}
function Fa(){var a=D,b;for(b in a)return!1;return!0}
function Ga(a,b){if(null!==a&&b in a)throw Error('The object already contains the key "'+b+'"');a[b]=!0}
function Ha(a){var b={},c;for(c in a)b[c]=a[c];return b}
var Ia="constructor hasOwnProperty isPrototypeOf propertyIsEnumerable toLocaleString toString valueOf".split(" ");function Ja(a,b){for(var c,d,e=1;e<arguments.length;e++){d=arguments[e];for(c in d)a[c]=d[c];for(var f=0;f<Ia.length;f++)c=Ia[f],Object.prototype.hasOwnProperty.call(d,c)&&(a[c]=d[c])}}
;function Ka(a){Ka[" "](a);return a}
Ka[" "]=t;function La(a,b){var c=Ma;return Object.prototype.hasOwnProperty.call(c,a)?c[a]:c[a]=b(a)}
;function Na(){var a=Oa;try{var b;if(b=!!a&&null!=a.location.href)a:{try{Ka(a.foo);b=!0;break a}catch(c){}b=!1}return b}catch(c){return!1}}
;var Pa=function(){var a=!1;try{var b=Object.defineProperty({},"passive",{get:function(){a=!0}});
n.addEventListener("test",null,b)}catch(c){}return a}();var Qa=!1,Ra="";function Sa(a){a=a.match(/[\d]+/g);if(!a)return"";a.length=3;return a.join(".")}
(function(){if(navigator.plugins&&navigator.plugins.length){var a=navigator.plugins["Shockwave Flash"];if(a&&(Qa=!0,a.description)){Ra=Sa(a.description);return}if(navigator.plugins["Shockwave Flash 2.0"]){Qa=!0;Ra="2.0.0.11";return}}if(navigator.mimeTypes&&navigator.mimeTypes.length&&(a=navigator.mimeTypes["application/x-shockwave-flash"],Qa=!(!a||!a.enabledPlugin))){Ra=Sa(a.enabledPlugin.description);return}try{var b=new ActiveXObject("ShockwaveFlash.ShockwaveFlash.7");Qa=!0;Ra=Sa(b.GetVariable("$version"));
return}catch(c){}try{b=new ActiveXObject("ShockwaveFlash.ShockwaveFlash.6");Qa=!0;Ra="6.0.21";return}catch(c){}try{b=new ActiveXObject("ShockwaveFlash.ShockwaveFlash"),Qa=!0,Ra=Sa(b.GetVariable("$version"))}catch(c){}})();
var Ta=Qa,Ua=Ra;var F;a:{var Va=n.navigator;if(Va){var Wa=Va.userAgent;if(Wa){F=Wa;break a}}F=""}function H(a){return-1!=F.indexOf(a)}
;function Xa(){return(H("Chrome")||H("CriOS"))&&!H("Edge")}
;function Ya(){return H("iPhone")&&!H("iPod")&&!H("iPad")}
;var Za=H("Opera"),I=H("Trident")||H("MSIE"),$a=H("Edge"),ab=H("Gecko")&&!(-1!=F.toLowerCase().indexOf("webkit")&&!H("Edge"))&&!(H("Trident")||H("MSIE"))&&!H("Edge"),bb=-1!=F.toLowerCase().indexOf("webkit")&&!H("Edge"),cb=H("Macintosh"),db=H("Windows"),eb=H("Android"),fb=Ya(),gb=H("iPad"),hb=H("iPod");function ib(){var a=n.document;return a?a.documentMode:void 0}
var jb;a:{var kb="",lb=function(){var a=F;if(ab)return/rv\:([^\);]+)(\)|;)/.exec(a);if($a)return/Edge\/([\d\.]+)/.exec(a);if(I)return/\b(?:MSIE|rv)[: ]([^\);]+)(\)|;)/.exec(a);if(bb)return/WebKit\/(\S+)/.exec(a);if(Za)return/(?:Version)[ \/]?(\S+)/.exec(a)}();
lb&&(kb=lb?lb[1]:"");if(I){var mb=ib();if(null!=mb&&mb>parseFloat(kb)){jb=String(mb);break a}}jb=kb}var nb=jb,Ma={};
function J(a){return La(a,function(){for(var b=0,c=ra(String(nb)).split("."),d=ra(String(a)).split("."),e=Math.max(c.length,d.length),f=0;0==b&&f<e;f++){var g=c[f]||"",h=d[f]||"";do{g=/(\d*)(\D*)(.*)/.exec(g)||["","","",""];h=/(\d*)(\D*)(.*)/.exec(h)||["","","",""];if(0==g[0].length&&0==h[0].length)break;b=sa(0==g[1].length?0:parseInt(g[1],10),0==h[1].length?0:parseInt(h[1],10))||sa(0==g[2].length,0==h[2].length)||sa(g[2],h[2]);g=g[3];h=h[3]}while(0==b)}return 0<=b})}
var ob;var pb=n.document;ob=pb&&I?ib()||("CSS1Compat"==pb.compatMode?parseInt(nb,10):5):void 0;(function(){var a;return db?(a=/Windows NT ([0-9.]+)/,(a=a.exec(F))?a[1]:"0"):cb?(a=/10[_.][0-9_.]+/,(a=a.exec(F))?a[0].replace(/_/g,"."):"10"):eb?(a=/Android\s+([^\);]+)(\)|;)/,(a=a.exec(F))?a[1]:""):fb||gb||hb?(a=/(?:iPhone|CPU)\s+OS\s+(\S+)/,(a=a.exec(F))?a[1].replace(/_/g,"."):""):""})();var qb=H("Firefox"),rb=Ya()||H("iPod"),sb=H("iPad"),tb=H("Android")&&!(Xa()||H("Firefox")||H("Opera")||H("Silk")),ub=Xa(),vb=H("Safari")&&!(Xa()||H("Coast")||H("Opera")||H("Edge")||H("Silk")||H("Android"))&&!(Ya()||H("iPad")||H("iPod"));function wb(a){return(a=a.exec(F))?a[1]:""}
(function(){if(qb)return wb(/Firefox\/([0-9.]+)/);if(I||$a||Za)return nb;if(ub)return Ya()||H("iPad")||H("iPod")?wb(/CriOS\/([0-9.]+)/):wb(/Chrome\/([0-9.]+)/);if(vb&&!(Ya()||H("iPad")||H("iPod")))return wb(/Version\/([0-9.]+)/);if(rb||sb){var a=/Version\/(\S+).*Mobile\/(\S+)/.exec(F);if(a)return a[1]+"."+a[2]}else if(tb)return(a=wb(/Android\s+([0-9.]+)/))?a:wb(/Version\/([0-9.]+)/);return""})();!ab&&!I||I&&9<=Number(ob)||ab&&J("1.9.1");I&&J("9");function xb(){this.b="";this.f=yb}
xb.prototype.la=!0;xb.prototype.ka=function(){return this.b};
var yb={};function zb(){this.b="";this.f=Ab}
zb.prototype.la=!0;zb.prototype.ka=function(){return this.b};
function Bb(a){return a instanceof zb&&a.constructor===zb&&a.f===Ab?a.b:"type_error:SafeUrl"}
var Cb=/^(?:(?:https?|mailto|ftp):|[^&:/?#]*(?:[/?#]|$))/i;function Db(a){if(a instanceof zb)return a;a=a.la?a.ka():String(a);Cb.test(a)||(a="about:invalid#zClosurez");return Eb(a)}
var Ab={};function Eb(a){var b=new zb;b.b=a;return b}
Eb("about:blank");function Fb(){this.b=""}
Fb.prototype.la=!0;Fb.prototype.ka=function(){return this.b};
function Gb(a){var b=new Fb;b.b=a;return b}
Gb("<!DOCTYPE html>");Gb("");Gb("<br>");function Hb(a,b){var c;c=b instanceof zb?b:Db(b);a.href=Bb(c)}
;function Ib(a){var b=document;return u(a)?b.getElementById(a):a}
function Jb(a){if(!a)return null;if(a.firstChild)return a.firstChild;for(;a&&!a.nextSibling;)a=a.parentNode;return a?a.nextSibling:null}
function Kb(a){if(!a)return null;if(!a.previousSibling)return a.parentNode;for(a=a.previousSibling;a&&a.lastChild;)a=a.lastChild;return a}
function Lb(a,b){for(var c=0;a;){if(b(a))return a;a=a.parentNode;c++}return null}
;function Mb(a){Nb();var b=new xb;b.b=a;return b}
var Nb=t;function Ob(a){"number"==typeof a&&(a=Math.round(a)+"px");return a}
;var Pb=/^(?:([^:/?#.]+):)?(?:\/\/(?:([^/?#]*)@)?([^/#?]*?)(?::([0-9]+))?(?=[/#?]|$))?([^?#]+)?(?:\?([^#]*))?(?:#([\s\S]*))?$/;function K(a){return a.match(Pb)}
function Qb(a){return a?decodeURI(a):a}
function Rb(a){if(a[1]){var b=a[0],c=b.indexOf("#");0<=c&&(a.push(b.substr(c)),a[0]=b=b.substr(0,c));c=b.indexOf("?");0>c?a[1]="?":c==b.length-1&&(a[1]=void 0)}return a.join("")}
function Sb(a,b,c){if(ia(b))for(var d=0;d<b.length;d++)Sb(a,String(b[d]),c);else null!=b&&c.push("&",a,""===b?"":"=",encodeURIComponent(String(b)))}
function Tb(a,b){for(var c in b)Sb(c,b[c],a);return a}
;var Ub=!!window.google_async_iframe_id,Oa=Ub&&window.parent||window;function Vb(a,b){var c=Wb();this.label=a;this.type=b;this.value=c;this.duration=0;this.uniqueId=this.label+"_"+this.type+"_"+Math.random()}
;function Xb(a,b){this.events=[];this.f=b||n;var c=null;b&&(b.google_js_reporting_queue=b.google_js_reporting_queue||[],this.events=b.google_js_reporting_queue,c=b.b);this.b=null!=c?c:Math.random()<a}
Xb.prototype.g=function(a){var b=this.f.performance;a&&b&&b.clearMarks&&(b.clearMarks("goog_"+a.uniqueId+"_start"),b.clearMarks("goog_"+a.uniqueId+"_end"))};
Xb.prototype.start=function(a,b){if(!this.b)return null;var c=new Vb(a,b),d=this.f.performance;d&&d.mark&&d.mark("goog_"+c.uniqueId+"_start");return c};
Xb.prototype.end=function(a){if(this.b){a.duration=Wb()-a.value;var b=this.f.performance;b&&b.mark&&b.mark("goog_"+a.uniqueId+"_end");this.b&&this.events.push(a)}};
function Wb(){var a=n.performance;return a&&a.now?a.now():x()}
;var Yb;if(Ub&&!Na()){var Zb="."+qa.domain;try{for(;2<Zb.split(".").length&&!Na();)qa.domain=Zb=Zb.substr(Zb.indexOf(".")+1),Oa=window.parent}catch(a){}Na()||(Oa=window)}Yb=Oa;var $b=new Xb(1,Yb);function ac(){Yb.b||(C($b.events,$b.g,$b),$b.events.length=0,$b.b=!1)}
if("complete"==Yb.document.readyState)ac();else if($b.b){var bc=function(){ac()};
Yb.addEventListener?Yb.addEventListener("load",bc,Pa?void 0:!1):Yb.attachEvent&&Yb.attachEvent("onload",bc)};var cc=(new Date).getTime();function dc(a){if(!a)return"";a=a.split("#")[0].split("?")[0];a=a.toLowerCase();0==a.indexOf("//")&&(a=window.location.protocol+a);/^[\w\-]*:\/\//.test(a)||(a=window.location.href);var b=a.substring(a.indexOf("://")+3),c=b.indexOf("/");-1!=c&&(b=b.substring(0,c));a=a.substring(0,a.indexOf("://"));if("http"!==a&&"https"!==a&&"chrome-extension"!==a&&"file"!==a&&"android-app"!==a&&"chrome-search"!==a)throw Error("Invalid URI scheme in origin");var c="",d=b.indexOf(":");if(-1!=d){var e=b.substring(d+
1),b=b.substring(0,d);if("http"===a&&"80"!==e||"https"===a&&"443"!==e)c=":"+e}return a+"://"+b+c}
;/*
 gapi.loader.OBJECT_CREATE_TEST_OVERRIDE &&*/
var ec=window,fc=document,gc=ec.location;function hc(){}
var ic=/\[native code\]/;function L(a,b,c){return a[b]=a[b]||c}
function jc(a){for(var b=0;b<this.length;b++)if(this[b]===a)return b;return-1}
function kc(a){a=a.sort();for(var b=[],c=void 0,d=0;d<a.length;d++){var e=a[d];e!=c&&b.push(e);c=e}return b}
function M(){var a;if((a=Object.create)&&ic.test(a))a=a(null);else{a={};for(var b in a)a[b]=void 0}return a}
var lc=L(ec,"gapi",{});var N;N=L(ec,"___jsl",M());L(N,"I",0);L(N,"hel",10);function mc(){var a=gc.href,b;if(N.dpo)b=N.h;else{b=N.h;var c=RegExp("([#].*&|[#])jsh=([^&#]*)","g"),d=RegExp("([?#].*&|[?#])jsh=([^&#]*)","g");if(a=a&&(c.exec(a)||d.exec(a)))try{b=decodeURIComponent(a[2])}catch(e){}}return b}
function nc(a){var b=L(N,"PQ",[]);N.PQ=[];var c=b.length;if(0===c)a();else for(var d=0,e=function(){++d===c&&a()},f=0;f<c;f++)b[f](e)}
function oc(a){return L(L(N,"H",M()),a,M())}
;function pc(){function a(){e[0]=1732584193;e[1]=4023233417;e[2]=2562383102;e[3]=271733878;e[4]=3285377520;y=m=0}
function b(a){for(var b=g,c=0;64>c;c+=4)b[c/4]=a[c]<<24|a[c+1]<<16|a[c+2]<<8|a[c+3];for(c=16;80>c;c++)a=b[c-3]^b[c-8]^b[c-14]^b[c-16],b[c]=(a<<1|a>>>31)&4294967295;a=e[0];for(var d=e[1],f=e[2],h=e[3],k=e[4],m,E,c=0;80>c;c++)40>c?20>c?(m=h^d&(f^h),E=1518500249):(m=d^f^h,E=1859775393):60>c?(m=d&f|h&(d|f),E=2400959708):(m=d^f^h,E=3395469782),m=((a<<5|a>>>27)&4294967295)+m+k+E+b[c]&4294967295,k=h,h=f,f=(d<<30|d>>>2)&4294967295,d=a,a=m;e[0]=e[0]+a&4294967295;e[1]=e[1]+d&4294967295;e[2]=e[2]+f&4294967295;
e[3]=e[3]+h&4294967295;e[4]=e[4]+k&4294967295}
function c(a,c){if("string"===typeof a){a=unescape(encodeURIComponent(a));for(var d=[],e=0,g=a.length;e<g;++e)d.push(a.charCodeAt(e));a=d}c||(c=a.length);d=0;if(0==m)for(;d+64<c;)b(a.slice(d,d+64)),d+=64,y+=64;for(;d<c;)if(f[m++]=a[d++],y++,64==m)for(m=0,b(f);d+64<c;)b(a.slice(d,d+64)),d+=64,y+=64}
function d(){var a=[],d=8*y;56>m?c(h,56-m):c(h,64-(m-56));for(var g=63;56<=g;g--)f[g]=d&255,d>>>=8;b(f);for(g=d=0;5>g;g++)for(var k=24;0<=k;k-=8)a[d++]=e[g]>>k&255;return a}
for(var e=[],f=[],g=[],h=[128],k=1;64>k;++k)h[k]=0;var m,y;a();return{reset:a,update:c,digest:d,Fa:function(){for(var a=d(),b="",c=0;c<a.length;c++)b+="0123456789ABCDEF".charAt(Math.floor(a[c]/16))+"0123456789ABCDEF".charAt(a[c]%16);return b}}}
;function qc(a,b,c){var d=[],e=[];if(1==(ia(c)?2:1))return e=[b,a],C(d,function(a){e.push(a)}),rc(e.join(" "));
var f=[],g=[];C(c,function(a){g.push(a.key);f.push(a.value)});
c=Math.floor((new Date).getTime()/1E3);e=0==f.length?[c,b,a]:[f.join(":"),c,b,a];C(d,function(a){e.push(a)});
a=rc(e.join(" "));a=[c,a];0==g.length||a.push(g.join(""));return a.join("_")}
function rc(a){var b=pc();b.update(a);return b.Fa().toLowerCase()}
;function sc(a){this.b=a||{cookie:""}}
l=sc.prototype;l.isEnabled=function(){return navigator.cookieEnabled};
l.set=function(a,b,c,d,e,f){if(/[;=\s]/.test(a))throw Error('Invalid cookie name "'+a+'"');if(/[;\r\n]/.test(b))throw Error('Invalid cookie value "'+b+'"');p(c)||(c=-1);e=e?";domain="+e:"";d=d?";path="+d:"";f=f?";secure":"";c=0>c?"":0==c?";expires="+(new Date(1970,1,1)).toUTCString():";expires="+(new Date(x()+1E3*c)).toUTCString();this.b.cookie=a+"="+b+e+d+c+f};
l.get=function(a,b){for(var c=a+"=",d=(this.b.cookie||"").split(";"),e=0,f;e<d.length;e++){f=ra(d[e]);if(0==f.lastIndexOf(c,0))return f.substr(c.length);if(f==a)return""}return b};
l.remove=function(a,b,c){var d=p(this.get(a));this.set(a,"",0,b,c);return d};
l.isEmpty=function(){return!this.b.cookie};
l.clear=function(){for(var a=(this.b.cookie||"").split(";"),b=[],c=[],d,e,f=0;f<a.length;f++)e=ra(a[f]),d=e.indexOf("="),-1==d?(b.push(""),c.push(e)):(b.push(e.substring(0,d)),c.push(e.substring(d+1)));for(a=b.length-1;0<=a;a--)this.remove(b[a])};
var tc=new sc("undefined"==typeof document?null:document);tc.f=3950;function uc(){var a=[],b=dc(String(n.location.href)),c=n.__OVERRIDE_SID;null==c&&(c=(new sc(document)).get("SID"));if(c&&(b=(c=0==b.indexOf("https:")||0==b.indexOf("chrome-extension:"))?n.__SAPISID:n.__APISID,null==b&&(b=(new sc(document)).get(c?"SAPISID":"APISID")),b)){var c=c?"SAPISIDHASH":"APISIDHASH",d=String(n.location.href);return d&&b&&c?[c,qc(dc(d),b,a||null)].join(" "):null}return null}
;var vc=L(N,"perf",M());L(vc,"g",M());var xc=L(vc,"i",M());L(vc,"r",[]);M();M();function yc(a,b,c){b&&0<b.length&&(b=zc(b),c&&0<c.length&&(b+="___"+zc(c)),28<b.length&&(b=b.substr(0,28)+(b.length-28)),c=b,b=L(xc,"_p",M()),L(b,c,M())[a]=(new Date).getTime(),b=vc.r,"function"===typeof b?b(a,"_p",c):b.push([a,"_p",c]))}
function zc(a){return a.join("__").replace(/\./g,"_").replace(/\-/g,"_").replace(/\,/g,"_")}
;var Ac=M(),Bc=[];function O(a){throw Error("Bad hint"+(a?": "+a:""));}
Bc.push(["jsl",function(a){for(var b in a)if(Object.prototype.hasOwnProperty.call(a,b)){var c=a[b];"object"==typeof c?N[b]=L(N,b,[]).concat(c):L(N,b,c)}if(b=a.u)a=L(N,"us",[]),a.push(b),(b=/^https:(.*)$/.exec(b))&&a.push("http:"+b[1])}]);
var Cc=/^(\/[a-zA-Z0-9_\-]+)+$/,Dc=[/\/amp\//,/\/amp$/,/^\/amp$/],Ec=/^[a-zA-Z0-9\-_\.,!]+$/,Fc=/^gapi\.loaded_[0-9]+$/,Gc=/^[a-zA-Z0-9,._-]+$/;function Hc(a,b,c,d){var e=a.split(";"),f=e.shift(),g=Ac[f],h=null;g?h=g(e,b,c,d):O("no hint processor for: "+f);h||O("failed to generate load url");b=h;c=b.match(Ic);(d=b.match(Jc))&&1===d.length&&Kc.test(b)&&c&&1===c.length||O("failed sanity: "+a);return h}
function Lc(a,b,c,d){function e(a){return encodeURIComponent(a).replace(/%2C/g,",")}
a=Mc(a);Fc.test(c)||O("invalid_callback");b=Nc(b);d=d&&d.length?Nc(d):null;return[encodeURIComponent(a.Za).replace(/%2C/g,",").replace(/%2F/g,"/"),"/k=",e(a.version),"/m=",e(b),d?"/exm="+e(d):"","/rt=j/sv=1/d=1/ed=1",a.qa?"/am="+e(a.qa):"",a.ya?"/rs="+e(a.ya):"",a.Ba?"/t="+e(a.Ba):"","/cb=",e(c)].join("")}
function Mc(a){"/"!==a.charAt(0)&&O("relative path");for(var b=a.substring(1).split("/"),c=[];b.length;){a=b.shift();if(!a.length||0==a.indexOf("."))O("empty/relative directory");else if(0<a.indexOf("=")){b.unshift(a);break}c.push(a)}a={};for(var d=0,e=b.length;d<e;++d){var f=b[d].split("="),g=decodeURIComponent(f[0]),h=decodeURIComponent(f[1]);2==f.length&&g&&h&&(a[g]=a[g]||h)}b="/"+c.join("/");Cc.test(b)||O("invalid_prefix");c=0;for(d=Dc.length;c<d;++c)Dc[c].test(b)&&O("invalid_prefix");c=Oc(a,
"k",!0);d=Oc(a,"am");e=Oc(a,"rs");a=Oc(a,"t");return{Za:b,version:c,qa:d,ya:e,Ba:a}}
function Nc(a){for(var b=[],c=0,d=a.length;c<d;++c){var e=a[c].replace(/\./g,"_").replace(/-/g,"_");Gc.test(e)&&b.push(e)}return b.join(",")}
function Oc(a,b,c){a=a[b];!a&&c&&O("missing: "+b);if(a){if(Ec.test(a))return a;O("invalid: "+b)}return null}
var Kc=/^https?:\/\/[a-z0-9_.-]+\.google\.com(:\d+)?\/[a-zA-Z0-9_.,!=\-\/]+$/,Jc=/\/cb=/g,Ic=/\/\//g;function Pc(){var a=mc();if(!a)throw Error("Bad hint");return a}
Ac.m=function(a,b,c,d){(a=a[0])||O("missing_hint");return"https://apis.google.com"+Lc(a,b,c,d)};
var Qc=decodeURI("%73cript"),Rc=/^[-+_0-9\/A-Za-z]+={0,2}$/;function Sc(a,b){for(var c=[],d=0;d<a.length;++d){var e=a[d];e&&0>jc.call(b,e)&&c.push(e)}return c}
function Tc(){var a=N.nonce;if(void 0!==a)return a&&a===String(a)&&a.match(Rc)?a:N.nonce=null;var b=L(N,"us",[]);if(!b||!b.length)return N.nonce=null;for(var c=fc.getElementsByTagName(Qc),d=0,e=c.length;d<e;++d){var f=c[d];if(f.src&&(a=String(f.nonce||f.getAttribute("nonce")||"")||null)){for(var g=0,h=b.length;g<h&&b[g]!==f.src;++g);if(g!==h&&a&&a===String(a)&&a.match(Rc))return N.nonce=a}}return null}
function Uc(a){if("loading"!=fc.readyState)Vc(a);else{var b=Tc(),c="";null!==b&&(c=' nonce="'+b+'"');fc.write("<"+Qc+' src="'+encodeURI(a)+'"'+c+"></"+Qc+">")}}
function Vc(a){var b=fc.createElement(Qc);b.setAttribute("src",a);a=Tc();null!==a&&b.setAttribute("nonce",a);b.async="true";(a=fc.getElementsByTagName(Qc)[0])?a.parentNode.insertBefore(b,a):(fc.head||fc.body||fc.documentElement).appendChild(b)}
function Wc(a,b){var c=b&&b._c;if(c)for(var d=0;d<Bc.length;d++){var e=Bc[d][0],f=Bc[d][1];f&&Object.prototype.hasOwnProperty.call(c,e)&&f(c[e],a,b)}}
function Xc(a,b,c){Yc(function(){var c;c=b===mc()?L(lc,"_",M()):M();c=L(oc(b),"_",c);a(c)},c)}
function Zc(a,b){var c=b||{};"function"==typeof b&&(c={},c.callback=b);Wc(a,c);var d=a?a.split(":"):[],e=c.h||Pc(),f=L(N,"ah",M());if(f["::"]&&d.length){for(var g=[],h=null;h=d.shift();){var k=h.split("."),k=f[h]||f[k[1]&&"ns:"+k[0]||""]||e,m=g.length&&g[g.length-1]||null,y=m;m&&m.hint==k||(y={hint:k,features:[]},g.push(y));y.features.push(h)}var E=g.length;if(1<E){var Z=c.callback;Z&&(c.callback=function(){0==--E&&Z()})}for(;d=g.shift();)$c(d.features,c,d.hint)}else $c(d||[],c,e)}
function $c(a,b,c){function d(a,b){if(E)return 0;ec.clearTimeout(y);Z.push.apply(Z,G);var d=((lc||{}).config||{}).update;d?d(f):f&&L(N,"cu",[]).push(f);if(b){yc("me0",a,ta);try{Xc(b,c,m)}finally{yc("me1",a,ta)}}return 1}
a=kc(a)||[];var e=b.callback,f=b.config,g=b.timeout,h=b.ontimeout,k=b.onerror,m=void 0;"function"==typeof k&&(m=k);var y=null,E=!1;if(g&&!h||!g&&h)throw"Timeout requires both the timeout parameter and ontimeout parameter to be set";var k=L(oc(c),"r",[]).sort(),Z=L(oc(c),"L",[]).sort(),ta=[].concat(k);0<g&&(y=ec.setTimeout(function(){E=!0;h()},g));
var G=Sc(a,Z);if(G.length){var G=Sc(a,k),fa=L(N,"CP",[]),ga=fa.length;fa[ga]=function(a){function b(){var a=fa[ga+1];a&&a()}
function c(b){fa[ga]=null;d(G,a)&&nc(function(){e&&e();b()})}
if(!a)return 0;yc("ml1",G,ta);0<ga&&fa[ga-1]?fa[ga]=function(){c(b)}:c(b)};
if(G.length){var wc="loaded_"+N.I++;lc[wc]=function(a){fa[ga](a);lc[wc]=null};
a=Hc(c,G,"gapi."+wc,k);k.push.apply(k,G);yc("ml0",G,ta);b.sync||ec.___gapisync?Uc(a):Vc(a)}else fa[ga](hc)}else d(G)&&e&&e()}
function Yc(a,b){if(N.hee&&0<N.hel)try{return a()}catch(c){b&&b(c),N.hel--,Zc("debug_error",function(){try{window.___jsl.hefn(c)}catch(d){throw c;}})}else try{return a()}catch(c){throw b&&b(c),c;
}}
lc.load=function(a,b){return Yc(function(){return Zc(a,b)})};function ad(a,b,c){this.i=c;this.g=a;this.j=b;this.f=0;this.b=null}
ad.prototype.get=function(){var a;0<this.f?(this.f--,a=this.b,this.b=a.next,a.next=null):a=this.g();return a};
function bd(a,b){a.j(b);a.f<a.i&&(a.f++,b.next=a.b,a.b=b)}
;function cd(a){n.setTimeout(function(){throw a;},0)}
var dd;
function ed(){var a=n.MessageChannel;"undefined"===typeof a&&"undefined"!==typeof window&&window.postMessage&&window.addEventListener&&!H("Presto")&&(a=function(){var a=document.createElement("IFRAME");a.style.display="none";a.src="";document.documentElement.appendChild(a);var b=a.contentWindow,a=b.document;a.open();a.write("");a.close();var c="callImmediate"+Math.random(),d="file:"==b.location.protocol?"*":b.location.protocol+"//"+b.location.host,a=v(function(a){if(("*"==d||a.origin==d)&&a.data==
c)this.port1.onmessage()},this);
b.addEventListener("message",a,!1);this.port1={};this.port2={postMessage:function(){b.postMessage(c,d)}}});
if("undefined"!==typeof a&&!H("Trident")&&!H("MSIE")){var b=new a,c={},d=c;b.port1.onmessage=function(){if(p(c.next)){c=c.next;var a=c.sa;c.sa=null;a()}};
return function(a){d.next={sa:a};d=d.next;b.port2.postMessage(0)}}return"undefined"!==typeof document&&"onreadystatechange"in document.createElement("SCRIPT")?function(a){var b=document.createElement("SCRIPT");
b.onreadystatechange=function(){b.onreadystatechange=null;b.parentNode.removeChild(b);b=null;a();a=null};
document.documentElement.appendChild(b)}:function(a){n.setTimeout(a,0)}}
;function fd(){this.f=this.b=null}
var hd=new ad(function(){return new gd},function(a){a.reset()},100);
fd.prototype.remove=function(){var a=null;this.b&&(a=this.b,this.b=this.b.next,this.b||(this.f=null),a.next=null);return a};
function gd(){this.next=this.scope=this.b=null}
gd.prototype.set=function(a,b){this.b=a;this.scope=b;this.next=null};
gd.prototype.reset=function(){this.next=this.scope=this.b=null};function id(a,b){jd||kd();ld||(jd(),ld=!0);var c=md,d=hd.get();d.set(a,b);c.f?c.f.next=d:c.b=d;c.f=d}
var jd;function kd(){if(-1!=String(n.Promise).indexOf("[native code]")){var a=n.Promise.resolve(void 0);jd=function(){a.then(nd)}}else jd=function(){var a=nd;
!ka(n.setImmediate)||n.Window&&n.Window.prototype&&!H("Edge")&&n.Window.prototype.setImmediate==n.setImmediate?(dd||(dd=ed()),dd(a)):n.setImmediate(a)}}
var ld=!1,md=new fd;function nd(){for(var a;a=md.remove();){try{a.b.call(a.scope)}catch(b){cd(b)}bd(hd,a)}ld=!1}
;var od="StopIteration"in n?n.StopIteration:{message:"StopIteration",stack:""};function pd(){}
pd.prototype.next=function(){throw od;};
pd.prototype.ca=function(){return this};
function qd(a){if(a instanceof pd)return a;if("function"==typeof a.ca)return a.ca(!1);if(ja(a)){var b=0,c=new pd;c.next=function(){for(;;){if(b>=a.length)throw od;if(b in a)return a[b++];b++}};
return c}throw Error("Not implemented");}
function rd(a,b){if(ja(a))try{C(a,b,void 0)}catch(c){if(c!==od)throw c;}else{a=qd(a);try{for(;;)b.call(void 0,a.next(),void 0,a)}catch(c){if(c!==od)throw c;}}}
function sd(a){if(ja(a))return za(a);a=qd(a);var b=[];rd(a,function(a){b.push(a)});
return b}
;function P(){this.f=this.f;this.F=this.F}
P.prototype.f=!1;P.prototype.dispose=function(){this.f||(this.f=!0,this.o())};
function td(a,b){a.f?p(void 0)?b.call(void 0):b():(a.F||(a.F=[]),a.F.push(p(void 0)?v(b,void 0):b))}
P.prototype.o=function(){if(this.F)for(;this.F.length;)this.F.shift()()};
function ud(a){a&&"function"==typeof a.dispose&&a.dispose()}
function vd(a){for(var b=0,c=arguments.length;b<c;++b){var d=arguments[b];ja(d)?vd.apply(null,d):ud(d)}}
;I&&J("9");!bb||J("528");ab&&J("1.9b")||I&&J("8")||Za&&J("9.5")||bb&&J("528");ab&&!J("8")||I&&J("9");function wd(a){return/^\s*$/.test(a)?!1:/^[\],:{}\s\u2028\u2029]*$/.test(a.replace(/\\["\\\/bfnrtu]/g,"@").replace(/(?:"[^"\\\n\r\u2028\u2029\x00-\x08\x0a-\x1f]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)[\s\u2028\u2029]*(?=:|,|]|}|$)/g,"]").replace(/(?:^|:|,)(?:[\s\u2028\u2029]*\[)+/g,""))}
function xd(a){a=String(a);if(wd(a))try{return eval("("+a+")")}catch(b){}throw Error("Invalid JSON string: "+a);}
function yd(a){return eval("("+a+")")}
function zd(a){var b=[];Ad(new Bd,a,b);return b.join("")}
function Bd(){}
function Ad(a,b,c){if(null==b)c.push("null");else{if("object"==typeof b){if(ia(b)){var d=b;b=d.length;c.push("[");for(var e="",f=0;f<b;f++)c.push(e),Ad(a,d[f],c),e=",";c.push("]");return}if(b instanceof String||b instanceof Number||b instanceof Boolean)b=b.valueOf();else{c.push("{");e="";for(d in b)Object.prototype.hasOwnProperty.call(b,d)&&(f=b[d],"function"!=typeof f&&(c.push(e),Cd(d,c),c.push(":"),Ad(a,f,c),e=","));c.push("}");return}}switch(typeof b){case "string":Cd(b,c);break;case "number":c.push(isFinite(b)&&
!isNaN(b)?String(b):"null");break;case "boolean":c.push(String(b));break;case "function":c.push("null");break;default:throw Error("Unknown type: "+typeof b);}}}
var Dd={'"':'\\"',"\\":"\\\\","/":"\\/","\b":"\\b","\f":"\\f","\n":"\\n","\r":"\\r","\t":"\\t","\x0B":"\\u000b"},Ed=/\uffff/.test("\uffff")?/[\\\"\x00-\x1f\x7f-\uffff]/g:/[\\\"\x00-\x1f\x7f-\xff]/g;function Cd(a,b){b.push('"',a.replace(Ed,function(a){var b=Dd[a];b||(b="\\u"+(a.charCodeAt(0)|65536).toString(16).substr(1),Dd[a]=b);return b}),'"')}
;function Fd(a){a.prototype.then=a.prototype.then;a.prototype.$goog_Thenable=!0}
;function Q(a,b){this.b=0;this.w=void 0;this.i=this.f=this.g=null;this.j=this.l=!1;if(a!=t)try{var c=this;a.call(b,function(a){Gd(c,2,a)},function(a){Gd(c,3,a)})}catch(d){Gd(this,3,d)}}
function Hd(){this.next=this.context=this.f=this.g=this.b=null;this.i=!1}
Hd.prototype.reset=function(){this.context=this.f=this.g=this.b=null;this.i=!1};
var Id=new ad(function(){return new Hd},function(a){a.reset()},100);
function Jd(a,b,c){var d=Id.get();d.g=a;d.f=b;d.context=c;return d}
function Kd(a){if(a instanceof Q)return a;var b=new Q(t);Gd(b,2,a);return b}
function Ld(a){return new Q(function(b,c){c(a)})}
Q.prototype.then=function(a,b,c){return Md(this,ka(a)?a:null,ka(b)?b:null,c)};
Fd(Q);Q.prototype.cancel=function(a){0==this.b&&id(function(){var b=new Nd(a);Od(this,b)},this)};
function Od(a,b){if(0==a.b)if(a.g){var c=a.g;if(c.f){for(var d=0,e=null,f=null,g=c.f;g&&(g.i||(d++,g.b==a&&(e=g),!(e&&1<d)));g=g.next)e||(f=g);e&&(0==c.b&&1==d?Od(c,b):(f?(d=f,d.next==c.i&&(c.i=d),d.next=d.next.next):Pd(c),Qd(c,e,3,b)))}a.g=null}else Gd(a,3,b)}
function Rd(a,b){a.f||2!=a.b&&3!=a.b||Sd(a);a.i?a.i.next=b:a.f=b;a.i=b}
function Md(a,b,c,d){var e=Jd(null,null,null);e.b=new Q(function(a,g){e.g=b?function(c){try{var e=b.call(d,c);a(e)}catch(m){g(m)}}:a;
e.f=c?function(b){try{var e=c.call(d,b);!p(e)&&b instanceof Nd?g(b):a(e)}catch(m){g(m)}}:g});
e.b.g=a;Rd(a,e);return e.b}
Q.prototype.C=function(a){this.b=0;Gd(this,2,a)};
Q.prototype.F=function(a){this.b=0;Gd(this,3,a)};
function Gd(a,b,c){if(0==a.b){a===c&&(b=3,c=new TypeError("Promise cannot resolve to itself"));a.b=1;var d;a:{var e=c,f=a.C,g=a.F;if(e instanceof Q)Rd(e,Jd(f||t,g||null,a)),d=!0;else{var h;if(e)try{h=!!e.$goog_Thenable}catch(m){h=!1}else h=!1;if(h)e.then(f,g,a),d=!0;else{if(la(e))try{var k=e.then;if(ka(k)){Td(e,k,f,g,a);d=!0;break a}}catch(m){g.call(a,m);d=!0;break a}d=!1}}}d||(a.w=c,a.b=b,a.g=null,Sd(a),3!=b||c instanceof Nd||Ud(a,c))}}
function Td(a,b,c,d,e){function f(a){h||(h=!0,d.call(e,a))}
function g(a){h||(h=!0,c.call(e,a))}
var h=!1;try{b.call(a,g,f)}catch(k){f(k)}}
function Sd(a){a.l||(a.l=!0,id(a.B,a))}
function Pd(a){var b=null;a.f&&(b=a.f,a.f=b.next,b.next=null);a.f||(a.i=null);return b}
Q.prototype.B=function(){for(var a;a=Pd(this);)Qd(this,a,this.b,this.w);this.l=!1};
function Qd(a,b,c,d){if(3==c&&b.f&&!b.i)for(;a&&a.j;a=a.g)a.j=!1;if(b.b)b.b.g=null,Vd(b,c,d);else try{b.i?b.g.call(b.context):Vd(b,c,d)}catch(e){Wd.call(null,e)}bd(Id,b)}
function Vd(a,b,c){2==b?a.g.call(a.context,c):a.f&&a.f.call(a.context,c)}
function Ud(a,b){a.j=!0;id(function(){a.j&&Wd.call(null,b)})}
var Wd=cd;function Nd(a){B.call(this,a)}
z(Nd,B);Nd.prototype.name="cancel";function R(a){P.call(this);this.l=1;this.i=[];this.j=0;this.b=[];this.g={};this.w=!!a}
z(R,P);l=R.prototype;l.subscribe=function(a,b,c){var d=this.g[a];d||(d=this.g[a]=[]);var e=this.l;this.b[e]=a;this.b[e+1]=b;this.b[e+2]=c;this.l=e+3;d.push(e);return e};
function Xd(a,b,c,d){if(b=a.g[b]){var e=a.b;(b=xa(b,function(a){return e[a+1]==c&&e[a+2]==d}))&&a.K(b)}}
l.K=function(a){var b=this.b[a];if(b){var c=this.g[b];0!=this.j?(this.i.push(a),this.b[a+1]=t):(c&&ya(c,a),delete this.b[a],delete this.b[a+1],delete this.b[a+2])}return!!b};
l.V=function(a,b){var c=this.g[a];if(c){for(var d=Array(arguments.length-1),e=1,f=arguments.length;e<f;e++)d[e-1]=arguments[e];if(this.w)for(e=0;e<c.length;e++){var g=c[e];Yd(this.b[g+1],this.b[g+2],d)}else{this.j++;try{for(e=0,f=c.length;e<f;e++)g=c[e],this.b[g+1].apply(this.b[g+2],d)}finally{if(this.j--,0<this.i.length&&0==this.j)for(;c=this.i.pop();)this.K(c)}}return 0!=e}return!1};
function Yd(a,b,c){id(function(){a.apply(b,c)})}
l.clear=function(a){if(a){var b=this.g[a];b&&(C(b,this.K,this),delete this.g[a])}else this.b.length=0,this.g={}};
l.o=function(){R.A.o.call(this);this.clear();this.i.length=0};function Zd(a){this.b=a}
Zd.prototype.set=function(a,b){p(b)?this.b.set(a,zd(b)):this.b.remove(a)};
Zd.prototype.get=function(a){var b;try{b=this.b.get(a)}catch(c){return}if(null!==b)try{return xd(b)}catch(c){throw"Storage: Invalid value was encountered";}};
Zd.prototype.remove=function(a){this.b.remove(a)};function $d(a){this.b=a}
z($d,Zd);function ae(a){this.data=a}
function be(a){return!p(a)||a instanceof ae?a:new ae(a)}
$d.prototype.set=function(a,b){$d.A.set.call(this,a,be(b))};
$d.prototype.f=function(a){a=$d.A.get.call(this,a);if(!p(a)||a instanceof Object)return a;throw"Storage: Invalid value was encountered";};
$d.prototype.get=function(a){if(a=this.f(a)){if(a=a.data,!p(a))throw"Storage: Invalid value was encountered";}else a=void 0;return a};function ce(a){this.b=a}
z(ce,$d);ce.prototype.set=function(a,b,c){if(b=be(b)){if(c){if(c<x()){ce.prototype.remove.call(this,a);return}b.expiration=c}b.creation=x()}ce.A.set.call(this,a,b)};
ce.prototype.f=function(a,b){var c=ce.A.f.call(this,a);if(c){var d;if(d=!b){d=c.creation;var e=c.expiration;d=!!e&&e<x()||!!d&&d>x()}if(d)ce.prototype.remove.call(this,a);else return c}};function de(a){this.b=a}
z(de,ce);function ee(){}
;function fe(){}
z(fe,ee);fe.prototype.clear=function(){var a=sd(this.ca(!0)),b=this;C(a,function(a){b.remove(a)})};function ge(a){this.b=a}
z(ge,fe);l=ge.prototype;l.isAvailable=function(){if(!this.b)return!1;try{return this.b.setItem("__sak","1"),this.b.removeItem("__sak"),!0}catch(a){return!1}};
l.set=function(a,b){try{this.b.setItem(a,b)}catch(c){if(0==this.b.length)throw"Storage mechanism: Storage disabled";throw"Storage mechanism: Quota exceeded";}};
l.get=function(a){a=this.b.getItem(a);if(!u(a)&&null!==a)throw"Storage mechanism: Invalid value was encountered";return a};
l.remove=function(a){this.b.removeItem(a)};
l.ca=function(a){var b=0,c=this.b,d=new pd;d.next=function(){if(b>=c.length)throw od;var d=c.key(b++);if(a)return d;d=c.getItem(d);if(!u(d))throw"Storage mechanism: Invalid value was encountered";return d};
return d};
l.clear=function(){this.b.clear()};
l.key=function(a){return this.b.key(a)};function he(){var a=null;try{a=window.localStorage||null}catch(b){}this.b=a}
z(he,ge);function ie(){var a=null;try{a=window.sessionStorage||null}catch(b){}this.b=a}
z(ie,ge);var je=window.performance&&window.performance.timing&&window.performance.now?function(){return window.performance.timing.navigationStart+window.performance.now()}:function(){return(new Date).getTime()},ke="Microsoft Internet Explorer"==navigator.appName;
function le(a,b){if(1<b.length)a[b[0]]=b[1];else{var c=b[0],d;for(d in c)a[d]=c[d]}}
;var me=window.yt&&window.yt.config_||window.ytcfg&&window.ytcfg.data_||{};q("yt.config_",me,void 0);function S(a){le(me,arguments)}
function T(a,b){return a in me?me[a]:b}
;function U(a){return T("EXPERIMENT_FLAGS",{})[a]}
;var ne={};function oe(a){return ne[a]||(ne[a]=String(a).replace(/\-([a-z])/g,function(a,c){return c.toUpperCase()}))}
function pe(a,b){return a?a.dataset?a.dataset[oe(b)]:a.getAttribute("data-"+b):null}
function qe(a){a&&(a.dataset?a.dataset[oe("loaded")]="true":a.setAttribute("data-loaded","true"))}
;function V(a,b){var c=r("yt.logging.errors.log");c?c(a,b,void 0,void 0,void 0):(c=T("ERRORS",[]),c.push([a,b,void 0,void 0,void 0]),S("ERRORS",c))}
function re(a){return a&&window.yterr?function(){try{return a.apply(this,arguments)}catch(b){V(b)}}:a}
;function W(a,b){ka(a)&&(a=re(a));return window.setTimeout(a,b)}
;var se=r("ytPubsubPubsubInstance")||r("yt.pubsub.instance_")||new R;R.prototype.subscribe=R.prototype.subscribe;R.prototype.unsubscribeByKey=R.prototype.K;R.prototype.publish=R.prototype.V;R.prototype.clear=R.prototype.clear;q("yt.pubsub.instance_",se,void 0);q("ytPubsubPubsubInstance",se,void 0);var te=r("ytPubsubPubsubSubscribedKeys")||r("yt.pubsub.subscribedKeys_")||{};q("yt.pubsub.subscribedKeys_",te,void 0);q("ytPubsubPubsubSubscribedKeys",te,void 0);
var ue=r("ytPubsubPubsubTopicToKeys")||r("yt.pubsub.topicToKeys_")||{};q("yt.pubsub.topicToKeys_",ue,void 0);q("ytPubsubPubsubTopicToKeys",ue,void 0);var ve=r("ytPubsubPubsubIsSynchronous")||r("yt.pubsub.isSynchronous_")||{};q("yt.pubsub.isSynchronous_",ve,void 0);q("ytPubsubPubsubIsSynchronous",ve,void 0);function we(){return r("ytPubsubPubsubInstance")||r("yt.pubsub.instance_")}
function xe(a){ue[a]&&(a=ue[a],C(a,function(a){te[a]&&delete te[a]}),a.length=0)}
function ye(a){var b=we();if(b)if(b.clear(a),a)xe(a);else for(var c in ue)xe(c)}
function ze(a,b){var c=we();c&&c.publish.apply(c,arguments)}
function Ae(a,b){var c=we();if(c){var d=c.subscribe(a,function(){var c=arguments,f;f=function(){te[d]&&b.apply(window,c)};
try{ve[a]?f():W(f,0)}catch(g){V(g)}},void 0);
te[d]=!0;ue[a]||(ue[a]=[]);ue[a].push(d);return d}return 0}
function Be(a){var b=we();b&&("number"==typeof a?a=[a]:u(a)&&(a=[parseInt(a,10)]),C(a,function(a){b.unsubscribeByKey(a);delete te[a]}))}
;var Ce=/\.vflset|-vfl[a-zA-Z0-9_+=-]+/,De=/-[a-zA-Z]{2,3}_[a-zA-Z]{2,3}(?=(\/|$))/;function Ee(a,b){var c=Fe(a),d=document.getElementById(c),e=d&&pe(d,"loaded"),f=d&&!e;if(e)b&&b();else{if(b){var e=Ae(c,b),g=""+(b[ma]||(b[ma]=++na));Ge[g]=e}f||(d=He(a,c,function(){pe(d,"loaded")||(qe(d),ze(c),W(w(ye,c),0))}))}}
function He(a,b,c){var d=document.createElement("script");d.id=b;d.onload=function(){c&&setTimeout(c,0)};
d.onreadystatechange=function(){switch(d.readyState){case "loaded":case "complete":d.onload()}};
d.src=a;a=document.getElementsByTagName("head")[0]||document.body;a.insertBefore(d,a.firstChild);return d}
function Ie(a,b){if(a&&b){var c=""+(b[ma]||(b[ma]=++na));(c=Ge[c])&&Be(c)}}
function Fe(a){var b=document.createElement("a");Hb(b,a);a=b.href.replace(/^[a-zA-Z]+:\/\//,"//");return"js-"+ua(a)}
var Ge={};function Je(a,b){if(window.spf){var c="";if(a){var d=a.indexOf("jsbin/"),e=a.lastIndexOf(".js"),f=d+6;-1<d&&-1<e&&e>f&&(c=a.substring(f,e),c=c.replace(Ce,""),c=c.replace(De,""),c=c.replace("debug-",""),c=c.replace("tracing-",""))}spf.script.load(a,c,b)}else Ee(a,b)}
;var Ke=null;function Le(){var a=T("BG_I",null),b=T("BG_IU",null),c=T("BG_P",void 0);b?Je(b,function(){Me(c)}):a&&(eval(a),Me(c))}
function Me(a){Ke=new botguard.bg(a);U("botguard_periodic_refresh")&&je()}
function Ne(){return null!=Ke}
function Oe(){return Ke?Ke.invoke():null}
;var Pe=p(XMLHttpRequest)?function(){return new XMLHttpRequest}:p(ActiveXObject)?function(){return new ActiveXObject("Microsoft.XMLHTTP")}:null;
function Qe(){if(!Pe)return null;var a=Pe();return"open"in a?a:null}
function Re(a){switch(a&&"status"in a?a.status:-1){case 200:case 201:case 202:case 203:case 204:case 205:case 206:case 304:return!0;default:return!1}}
;function Se(a){"?"==a.charAt(0)&&(a=a.substr(1));a=a.split("&");for(var b={},c=0,d=a.length;c<d;c++){var e=a[c].split("=");if(1==e.length&&e[0]||2==e.length){var f=decodeURIComponent((e[0]||"").replace(/\+/g," ")),e=decodeURIComponent((e[1]||"").replace(/\+/g," "));f in b?ia(b[f])?Aa(b[f],e):b[f]=[b[f],e]:b[f]=e}}return b}
function Te(a,b){var c=a.split("#",2);a=c[0];var c=1<c.length?"#"+c[1]:"",d=a.split("?",2);a=d[0];var d=Se(d[1]||""),e;for(e in b)d[e]=b[e];return Rb(Tb([a],d))+c}
function Ue(a){a=Tb([],a);a[0]="";return a.join("")}
;var Ve={"X-Goog-Visitor-Id":"SANDBOXED_VISITOR_ID","X-YouTube-Client-Name":"INNERTUBE_CONTEXT_CLIENT_NAME","X-YouTube-Client-Version":"INNERTUBE_CONTEXT_CLIENT_VERSION","X-Youtube-Identity-Token":"ID_TOKEN","X-YouTube-Page-CL":"PAGE_CL","X-YouTube-Page-Label":"PAGE_BUILD_LABEL","X-YouTube-Variants-Checksum":"VARIANTS_CHECKSUM"};
function We(a,b){b=void 0===b?{}:b;var c=void 0;c=window.location.href;var d=K(a)[1]||null,e=Qb(K(a)[3]||null);d&&e?(d=c,c=K(a),d=K(d),c=c[3]==d[3]&&c[1]==d[1]&&c[4]==d[4]):c=e?Qb(K(c)[3]||null)==e&&(Number(K(c)[4]||null)||null)==(Number(K(a)[4]||null)||null):!0;for(var f in Ve){if((e=d=T(Ve[f]))&&!(e=c)){var g=a,e=f,h=T("CORS_HEADER_WHITELIST")||{};e=(g=Qb(K(g)[3]||null))?(h=h[g])?0<=va(h,e):!1:!0}e&&(b[f]=d)}return b}
function Xe(a,b){var c=T("XSRF_FIELD_NAME",void 0),d;b.headers&&(d=b.headers["Content-Type"]);return!b.qb&&(!Qb(K(a)[3]||null)||b.withCredentials||Qb(K(a)[3]||null)==document.location.hostname)&&"POST"==b.method&&(!d||"application/x-www-form-urlencoded"==d)&&!(b.D&&b.D[c])}
function Ye(a,b){var c=b.format||"JSON";b.Ka&&(a=document.location.protocol+"//"+document.location.hostname+(document.location.port?":"+document.location.port:"")+a);var d=T("XSRF_FIELD_NAME",void 0),e=T("XSRF_TOKEN",void 0),f=b.kb;f&&(f[d]&&delete f[d],a=Te(a,f||{}));var g=b.postBody||"",f=b.D;Xe(a,b)&&(f||(f={}),f[d]=e);f&&u(g)&&(d=Se(g),Ja(d,f),g=b.wa&&"JSON"==b.wa?JSON.stringify(d):Ue(d));var h=!1,k,m=Ze(a,function(a){if(!h){h=!0;k&&window.clearTimeout(k);var d=Re(a),e=null;if(d||400<=a.status&&
500>a.status)e=$e(c,a,b.pb);if(d)a:if(204==a.status)d=!0;else{switch(c){case "XML":d=0==parseInt(e&&e.return_code,10);break a;case "RAW":d=!0;break a}d=!!e}var e=e||{},f=b.context||n;d?b.J&&b.J.call(f,a,e):b.onError&&b.onError.call(f,a,e);b.Qa&&b.Qa.call(f,a,e)}},b.method,g,b.headers,b.responseType,b.withCredentials);
b.O&&0<b.timeout&&(k=W(function(){h||(h=!0,m.abort(),window.clearTimeout(k),b.O.call(b.context||n,m))},b.timeout))}
function $e(a,b,c){var d=null;switch(a){case "JSON":a=b.responseText;b=b.getResponseHeader("Content-Type")||"";a&&0<=b.indexOf("json")&&(d=yd(a));break;case "XML":if(b=(b=b.responseXML)?af(b):null)d={},C(b.getElementsByTagName("*"),function(a){d[a.tagName]=bf(a)})}c&&cf(d);
return d}
function cf(a){if(la(a))for(var b in a){var c;(c="html_content"==b)||(c=b.length-5,c=0<=c&&b.indexOf("_html",c)==c);if(c){c=b;var d;d=Gb(a[b]);a[c]=d}else cf(a[b])}}
function af(a){return a?(a=("responseXML"in a?a.responseXML:a).getElementsByTagName("root"))&&0<a.length?a[0]:null:null}
function bf(a){var b="";C(a.childNodes,function(a){b+=a.nodeValue});
return b}
function df(a,b){b.method="POST";b.D||(b.D={});Ye(a,b)}
function Ze(a,b,c,d,e,f,g){function h(){4==(k&&"readyState"in k?k.readyState:0)&&b&&re(b)(k)}
c=void 0===c?"GET":c;d=void 0===d?"":d;var k=Qe();if(!k)return null;"onloadend"in k?k.addEventListener("loadend",h,!1):k.onreadystatechange=h;k.open(c,a,!0);f&&(k.responseType=f);g&&(k.withCredentials=!0);c="POST"==c;if(e=We(a,e))for(var m in e)k.setRequestHeader(m,e[m]),"content-type"==m.toLowerCase()&&(c=!1);c&&k.setRequestHeader("Content-Type","application/x-www-form-urlencoded");k.send(d);return k}
;var ef={},ff=0;function gf(a,b){a&&(T("USE_NET_AJAX_FOR_PING_TRANSPORT",!1)?Ze(a,b):hf(a,b))}
function hf(a,b){var c=new Image,d=""+ff++;ef[d]=c;c.onload=c.onerror=function(){b&&ef[d]&&b();delete ef[d]};
c.src=a}
;function jf(a,b,c,d,e){c={name:c||T("INNERTUBE_CONTEXT_CLIENT_NAME",1),version:d||T("INNERTUBE_CONTEXT_CLIENT_VERSION",void 0)};e=window&&window.yterr||e||!1;if(a&&e&&!(5<=kf)){e=a.stacktrace;d=a.columnNumber;var f=r("window.location.href");if(u(a))a={message:a,name:"Unknown error",lineNumber:"Not available",fileName:f,stack:"Not available"};else{var g,h,k=!1;try{g=a.lineNumber||a.line||"Not available"}catch(E){g="Not available",k=!0}try{h=a.fileName||a.filename||a.sourceURL||n.$googDebugFname||f}catch(E){h=
"Not available",k=!0}a=!k&&a.lineNumber&&a.fileName&&a.stack&&a.message&&a.name?a:{message:a.message||"Not available",name:a.name||"UnknownError",lineNumber:g,fileName:h,stack:a.stack||"Not available"}}e=e||a.stack;g=a.lineNumber.toString();isNaN(g)||isNaN(d)||(g=g+":"+d);if(!(lf[a.message]||0<=e.indexOf("/YouTubeCenter.js")||0<=e.indexOf("/mytube.js"))){h=a.fileName;b={kb:{a:"logerror",t:"jserror",type:a.name,msg:a.message.substr(0,1E3),line:g,level:b||"ERROR"},D:{url:T("PAGE_NAME",window.location.href),
file:h},method:"POST"};e&&(b.D.stack=e);for(var m in c)b.D["client."+m]=c[m];if(m=T("LATEST_ECATCHER_SERVICE_TRACKING_PARAMS",void 0))for(var y in m)b.D[y]=m[y];Ye("/error_204",b);lf[a.message]=!0;kf++}}}
var lf={},kf=0;var mf=window.yt&&window.yt.msgs_||window.ytcfg&&window.ytcfg.msgs||{};q("yt.msgs_",mf,void 0);function nf(a){le(mf,arguments)}
;function of(a,b){var c=5E3;isNaN(c)&&(c=void 0);var d=r("yt.scheduler.instance.addJob");return d?d(a,b,c):void 0===c?(a(),NaN):W(a,c||0)}
;var pf=[],qf=!1;function rf(){function a(){qf=!0;"google_ad_status"in window?S("DCLKSTAT",1):S("DCLKSTAT",2)}
Je("//static.doubleclick.net/instream/ad_status.js",a);pf.push(of(function(){qf||"google_ad_status"in window||(Ie("//static.doubleclick.net/instream/ad_status.js",a),S("DCLKSTAT",3))},1))}
function sf(){return parseInt(T("DCLKSTAT",0),10)}
;var tf=0,uf=r("ytDomDomGetNextId")||function(){return++tf};
q("ytDomDomGetNextId",uf,void 0);var vf={stopImmediatePropagation:1,stopPropagation:1,preventMouseEvent:1,preventManipulation:1,preventDefault:1,layerX:1,layerY:1,screenX:1,screenY:1,scale:1,rotation:1,webkitMovementX:1,webkitMovementY:1};
function wf(a){this.type="";this.state=this.source=this.data=this.currentTarget=this.relatedTarget=this.target=null;this.charCode=this.keyCode=0;this.shiftKey=this.ctrlKey=this.altKey=!1;this.clientY=this.clientX=0;this.changedTouches=this.touches=null;if(a=a||window.event){this.event=a;for(var b in a)b in vf||(this[b]=a[b]);(b=a.target||a.srcElement)&&3==b.nodeType&&(b=b.parentNode);this.target=b;if(b=a.relatedTarget)try{b=b.nodeName?b:null}catch(c){b=null}else"mouseover"==this.type?b=a.fromElement:
"mouseout"==this.type&&(b=a.toElement);this.relatedTarget=b;this.clientX=void 0!=a.clientX?a.clientX:a.pageX;this.clientY=void 0!=a.clientY?a.clientY:a.pageY;this.keyCode=a.keyCode?a.keyCode:a.which;this.charCode=a.charCode||("keypress"==this.type?this.keyCode:0);this.altKey=a.altKey;this.ctrlKey=a.ctrlKey;this.shiftKey=a.shiftKey}}
wf.prototype.preventDefault=function(){this.event&&(this.event.returnValue=!1,this.event.preventDefault&&this.event.preventDefault())};
wf.prototype.stopPropagation=function(){this.event&&(this.event.cancelBubble=!0,this.event.stopPropagation&&this.event.stopPropagation())};
wf.prototype.stopImmediatePropagation=function(){this.event&&(this.event.cancelBubble=!0,this.event.stopImmediatePropagation&&this.event.stopImmediatePropagation())};var Ea=r("ytEventsEventsListeners")||{};q("ytEventsEventsListeners",Ea,void 0);var xf=r("ytEventsEventsCounter")||{count:0};q("ytEventsEventsCounter",xf,void 0);function yf(a,b,c,d){d=void 0===d?!1:d;a.addEventListener&&("mouseenter"!=b||"onmouseenter"in document?"mouseleave"!=b||"onmouseenter"in document?"mousewheel"==b&&"MozBoxSizing"in document.documentElement.style&&(b="MozMousePixelScroll"):b="mouseout":b="mouseover");return Da(function(e){return e[0]==a&&e[1]==b&&e[2]==c&&e[4]==!!d})}
function zf(a,b,c){var d;d=void 0===d?!1:d;if(!a||!a.addEventListener&&!a.attachEvent)return"";var e=yf(a,b,c,d);if(e)return e;var e=++xf.count+"",f=!("mouseenter"!=b&&"mouseleave"!=b||!a.addEventListener||"onmouseenter"in document),g;g=f?function(d){d=new wf(d);if(!Lb(d.relatedTarget,function(b){return b==a}))return d.currentTarget=a,d.type=b,c.call(a,d)}:function(b){b=new wf(b);
b.currentTarget=a;return c.call(a,b)};
g=re(g);a.addEventListener?("mouseenter"==b&&f?b="mouseover":"mouseleave"==b&&f?b="mouseout":"mousewheel"==b&&"MozBoxSizing"in document.documentElement.style&&(b="MozMousePixelScroll"),a.addEventListener(b,g,d)):a.attachEvent("on"+b,g);Ea[e]=[a,b,c,g,d];return e}
function Af(a){a&&("string"==typeof a&&(a=[a]),C(a,function(a){if(a in Ea){var b=Ea[a],d=b[0],e=b[1],f=b[3],b=b[4];d.removeEventListener?d.removeEventListener(e,f,b):d.detachEvent&&d.detachEvent("on"+e,f);delete Ea[a]}}))}
;function Bf(){if(null==r("_lact",window)){var a=parseInt(T("LACT"),10),a=isFinite(a)?x()-Math.max(a,0):-1;q("_lact",a,window);q("_fact",a,window);-1==a&&Cf();zf(document,"keydown",Cf);zf(document,"keyup",Cf);zf(document,"mousedown",Cf);zf(document,"mouseup",Cf);Ae("page-mouse",Cf);Ae("page-scroll",Cf);Ae("page-resize",Cf)}}
function Cf(){null==r("_lact",window)&&(Bf(),r("_lact",window));var a=x();q("_lact",a,window);-1==r("_fact",window)&&q("_fact",a,window);ze("USER_ACTIVE")}
function Df(){var a=r("_lact",window);return null==a?-1:Math.max(x()-a,0)}
;function Ef(a,b,c,d){this.f=a;this.i=b;this.g=c;this.b=d}
var Ff=1;function Gf(a){var b={};void 0!==a.f?b.trackingParams=a.f:(b.veType=a.i,null!=a.g&&(b.veCounter=a.g));void 0!==a.b&&(b.dataElement=Gf(a.b));return b}
;var Hf={log_event:"events",log_interaction:"interactions"},If={},Jf={},Kf=0,D=r("ytLoggingTransportLogPayloadsQueue_")||{};q("ytLoggingTransportLogPayloadsQueue_",D,void 0);var Lf=r("ytLoggingTransportTokensToCttTargetIds_")||{};q("ytLoggingTransportTokensToCttTargetIds_",Lf,void 0);
function Mf(a,b){Jf[a.endpoint]=b;var c;if(a.da){c=a.da;var d={};c.videoId?d.videoId=c.videoId:c.playlistId&&(d.playlistId=c.playlistId);Lf[a.da.token]=d;c=Nf(a.endpoint,a.da.token)}else c=Nf(a.endpoint);c.push(a.va);d=Number(U("web_logging_max_batch")||0)||20;c.length>=d?Of():Pf()}
function Of(){window.clearTimeout(Kf);if(!Fa()){for(var a in D){var b=If[a];if(!b){var c=Jf[a];if(!c)continue;b=new c;If[a]=b}var c=void 0,d=a,e=Hf[d];for(c in D[d]){var f=b.f();f[e]=Nf(d,c);f.requestTimeMs=Math.round(je());var g=Lf[c];if(g)a:{var h,k=f,m=c;if(g.videoId)h="VIDEO";else if(g.playlistId)h="PLAYLIST";else break a;k.credentialTransferTokenTargetId=g;k.context=k.context||{};k.context.user=k.context.user||{};k.context.user.credentialTransferTokens=[{token:m,scope:h}]}delete Lf[c];b.g(d,
f,{})}delete D[a]}Fa()||Pf()}}
function Pf(){window.clearTimeout(Kf);Kf=W(Of,T("LOGGING_BATCH_TIMEOUT",1E4))}
function Nf(a,b){b||(b="");D[a]=D[a]||{};D[a][b]=D[a][b]||[];return D[a][b]}
;function Qf(a,b,c,d){Rf(a,{attachChild:{csn:b,parentVisualElement:Gf(c),visualElements:[Gf(d)]}},void 0)}
function Sf(a,b,c){c=wa(c,function(a){return Gf(a)});
Rf(a,{visibilityUpdate:{csn:b,visualElements:c}})}
function Rf(a,b,c){b.eventTimeMs=Math.round(je());b.lactMs=Df();c&&(b.clientData=Tf(c));Mf({endpoint:"log_interaction",va:b},a)}
function Tf(a){var b={};a.analyticsChannelData&&(b.analyticsDatas=wa(a.analyticsChannelData,function(a){return{tabName:a.tabName,cardName:a.cardName,isChannelScreen:a.isChannelScreen,insightId:a.insightId,externalChannelId:a.externalChannelId,externalContentOwnerId:a.externalContentOwnerId}}));
return{playbackData:{clientPlaybackNonce:a.clientPlaybackNonce},analyticsChannelData:b,externalLinkData:a.externalLinkData}}
;function Uf(){if(!Vf&&!Wf||!window.JSON)return null;var a;try{a=Vf.get("yt-player-two-stage-token")}catch(b){}if(!u(a))try{a=Wf.get("yt-player-two-stage-token")}catch(b){}if(!u(a))return null;try{a=JSON.parse(a,void 0)}catch(b){}return a}
var Wf,Xf=new he;Wf=Xf.isAvailable()?new de(Xf):null;var Vf,Yf=new ie;Vf=Yf.isAvailable()?new de(Yf):null;function Zf(){var a=T("ROOT_VE_TYPE",void 0);return a?new Ef(void 0,a,void 0):null}
function $f(){var a=T("client-screen-nonce",void 0);a||(a=T("EVENT_ID",void 0));return a}
;function ag(a,b,c){b=void 0===b?{}:b;c=void 0===c?!1:c;var d=T("EVENT_ID");d&&(b.ei||(b.ei=d));if(b){var d=a,e=T("VALID_SESSION_TEMPDATA_DOMAINS",[]),f=Qb(K(window.location.href)[3]||null);f&&e.push(f);f=Qb(K(d)[3]||null);if(0<=va(e,f)||!f&&0==d.lastIndexOf("/",0))if(U("autoescape_tempdata_url")&&(e=document.createElement("a"),Hb(e,d),d=e.href),d){var f=K(d),d=f[5],e=f[6],f=f[7],g="";d&&(g+=d);e&&(g+="?"+e);f&&(g+="#"+f);d=g;e=d.indexOf("#");if(d=0>e?d:d.substr(0,e)){if(b.itct||b.ved)b.csn=b.csn||
$f();d="ST-"+ua(d).toString(36);e=b?Ue(b):"";tc.set(""+d,e,5,"/","youtube.com");b&&(b=b.itct||b.ved,d=r("yt.logging.screen.storeParentElement"),b&&d&&d(new Ef(b)))}}}if(c)return!1;if((window.ytspf||{}).enabled)spf.navigate(a);else{var h,k,m;h=void 0===h?{}:h;k=void 0===k?"":k;m=void 0===m?window:m;c=m.location;a=Rb(Tb([a],h))+k;a=a instanceof zb?a:Db(a);c.href=Bb(a)}return!0}
;var bg=r("yt.abuse.botguardInitialized")||Ne;q("yt.abuse.botguardInitialized",bg,void 0);var cg=r("yt.abuse.invokeBotguard")||Oe;q("yt.abuse.invokeBotguard",cg,void 0);var dg=r("yt.abuse.dclkstatus.checkDclkStatus")||sf;q("yt.abuse.dclkstatus.checkDclkStatus",dg,void 0);var eg=r("yt.player.exports.navigate")||ag;q("yt.player.exports.navigate",eg,void 0);var fg=r("yt.util.activity.init")||Bf;q("yt.util.activity.init",fg,void 0);var gg=r("yt.util.activity.getTimeSinceActive")||Df;
q("yt.util.activity.getTimeSinceActive",gg,void 0);var hg=r("yt.util.activity.setTimestamp")||Cf;q("yt.util.activity.setTimestamp",hg,void 0);function ig(a,b,c,d,e){var f={};f.eventTimeMs=Math.round(d||je());f[a]=b;f.context={lastActivityMs:String(Df())};Mf({endpoint:"log_event",va:f,da:e},c)}
;function jg(a){a={client:{hl:a.Na,gl:a.Ma,clientName:a.La,clientVersion:a.innertubeContextClientVersion}};T("DELEGATED_SESSION_ID")&&(a.user={onBehalfOfUser:T("DELEGATED_SESSION_ID")});return a}
function kg(){return{apiaryHost:T("APIARY_HOST",void 0),Da:T("APIARY_HOST_FIRSTPARTY",void 0),gapiHintOverride:!!T("GAPI_HINT_OVERRIDE",void 0),gapiHintParams:T("GAPI_HINT_PARAMS",void 0),innertubeApiKey:T("INNERTUBE_API_KEY",void 0),innertubeApiVersion:T("INNERTUBE_API_VERSION",void 0),La:T("INNERTUBE_CONTEXT_CLIENT_NAME","WEB"),innertubeContextClientVersion:T("INNERTUBE_CONTEXT_CLIENT_VERSION",void 0),Na:T("INNERTUBE_CONTEXT_HL",void 0),Ma:T("INNERTUBE_CONTEXT_GL",void 0),xhrApiaryHost:T("XHR_APIARY_HOST",
void 0),Oa:T("INNERTUBE_HOST_OVERRIDE",void 0)}}
function lg(a,b,c){c.context&&c.context.capabilities&&(c=c.context.capabilities,c.snapshot||c.golden)&&(a="vix");return"/youtubei/"+a+"/"+b}
;function mg(a){this.b=a||kg();ng||(ng=og(this.b))}
function og(a){return(new Q(function(b){try{var c={gapiHintOverride:a.gapiHintOverride,_c:{jsl:{h:a.gapiHintParams}},callback:b},d=ka(c)?{callback:c}:c||{};d._c&&d._c.jsl&&d._c.jsl.h||Ja(d,{_c:{jsl:{h:T("GAPI_HINT_PARAMS",void 0)}}});if(d.gapiHintOverride||T("GAPI_HINT_OVERRIDE")){var e;var f=document.location.href;if(-1!=f.indexOf("?")){var f=(f||"").split("#")[0],g=f.split("?",2);e=Se(1<g.length?g[1]:g[0])}else e={};var h=e.gapi_jsh;h&&Ja(d,{_c:{jsl:{h:h}}})}Zc("client",d)}catch(k){V(k)}})).then(function(){})}
mg.prototype.i=function(){var a=r("gapi.config.update");a("googleapis.config/auth/useFirstPartyAuth",!0);a("googleapis.config/auth/useFirstPartyAuthV2",!0);var b=this.b.apiaryHost;/^[\s\xa0]*$/.test(null==b?"":String(b))||a("googleapis.config/root",(-1==b.indexOf("://")?"//":"")+b);b=this.b.Da;/^[\s\xa0]*$/.test(null==b?"":String(b))||a("googleapis.config/root-1p",(-1==b.indexOf("://")?"//":"")+b);b=T("SESSION_INDEX");a("googleapis.config/sessionIndex",b);r("gapi.client.setApiKey")(this.b.innertubeApiKey)};
mg.prototype.f=function(){return{context:jg(this.b)}};
mg.prototype.g=function(a,b,c){var d,e=!1;0<c.timeout&&(d=W(function(){e||(e=!0,c.O&&c.O())},c.timeout));
pg(this,a,b,function(a){if(!e)if(e=!0,d&&window.clearTimeout(d),a)c.J&&c.J(a);else if(c.onError)c.onError()})};
function pg(a,b,c,d){var e={path:lg(a.b.innertubeApiVersion,b,c),headers:{"X-Goog-Visitor-Id":T("VISITOR_DATA")},method:"POST",body:zd(c)},f=v(a.i,a);ng.then(function(){f();r("gapi.client.request")(e).execute(d||t)})}
var ng=null;function qg(a){this.b=a||kg()}
qg.prototype.f=function(){return{context:jg(this.b)}};
qg.prototype.g=function(a,b,c){T("VISITOR_DATA")||V(Error("Missing VISITOR_DATA when sending innertube request."));var d={headers:{"Content-Type":"application/json","X-Goog-Visitor-Id":T("VISITOR_DATA","")},D:b,wa:"JSON",O:c.O,J:function(a,b){c.J&&c.J(b)},
onError:function(a,b){if(c.onError)c.onError(b)},
timeout:c.timeout,withCredentials:!0},e=uc();e&&(d.headers.Authorization=e,d.headers["X-Goog-AuthUser"]=T("SESSION_INDEX",0));var f=this.b.xhrApiaryHost;f.startsWith("http")||(f="//"+f);if(U("youtubei_for_web")||U("unplugged_web_same_op_domain"))f="";var g=this.b.Oa;g&&(f=g);e&&!f&&(d.headers["x-origin"]=window.location.origin);df(""+f+lg(this.b.innertubeApiVersion,a,b)+"?alt=json&key="+this.b.innertubeApiKey,d)};function rg(){return U("enable_youtubei_innertube")?qg:mg}
;function sg(a){a=a||{};this.url=a.url||"";this.urlV9As2=a.url_v9as2||"";this.args=a.args||Ha(tg);this.assets=a.assets||{};this.attrs=a.attrs||Ha(ug);this.params=a.params||Ha(vg);this.minVersion=a.min_version||"8.0.0";this.fallback=a.fallback||null;this.fallbackMessage=a.fallbackMessage||null;this.html5=!!a.html5;this.disable=a.disable||{};this.loaded=!!a.loaded;this.messages=a.messages||{}}
var tg={enablejsapi:1},ug={},vg={allowscriptaccess:"always",allowfullscreen:"true",bgcolor:"#000000"};function wg(a){a instanceof sg||(a=new sg(a));return a}
function xg(a){var b=new sg,c;for(c in a)if(a.hasOwnProperty(c)){var d=a[c];b[c]="object"==ha(d)?Ha(d):d}return b}
;function yg(){this.g=this.f=this.b=0;this.i="";var a=r("window.navigator.plugins"),b=r("window.navigator.mimeTypes"),a=a&&a["Shockwave Flash"],b=b&&b["application/x-shockwave-flash"],b=a&&b&&b.enabledPlugin&&a.description||"";if(a=b){var c=a.indexOf("Shockwave Flash");0<=c&&(a=a.substr(c+15));for(var c=a.split(" "),d="",a="",e=0,f=c.length;e<f;e++)if(d)if(a)break;else a=c[e];else d=c[e];d=d.split(".");c=parseInt(d[0],10)||0;d=parseInt(d[1],10)||0;e=0;if("r"==a.charAt(0)||"d"==a.charAt(0))e=parseInt(a.substr(1),
10)||0;a=[c,d,e]}else a=[0,0,0];this.i=b;b=a;this.b=b[0];this.f=b[1];this.g=b[2];if(0>=this.b){var g,h,k,m;if(ke)try{g=new ActiveXObject("ShockwaveFlash.ShockwaveFlash")}catch(y){g=null}else k=document.body,m=document.createElement("object"),m.setAttribute("type","application/x-shockwave-flash"),g=k.appendChild(m);if(g&&"GetVariable"in g)try{h=g.GetVariable("$version")}catch(y){h=""}k&&m&&k.removeChild(m);(g=h||"")?(g=g.split(" ")[1].split(","),g=[parseInt(g[0],10)||0,parseInt(g[1],10)||0,parseInt(g[2],
10)||0]):g=[0,0,0];this.b=g[0];this.f=g[1];this.g=g[2]}}
ea(yg);function zg(a,b,c,d){b="string"==typeof b?b.split("."):[b,c,d];b[0]=parseInt(b[0],10)||0;b[1]=parseInt(b[1],10)||0;b[2]=parseInt(b[2],10)||0;return a.b>b[0]||a.b==b[0]&&a.f>b[1]||a.b==b[0]&&a.f==b[1]&&a.g>=b[2]}
;var Ag=/cssbin\/(?:debug-)?([a-zA-Z0-9_-]+?)(?:-2x|-web|-rtl|-vfl|.css)/;function Bg(a){if(window.spf){var b=a.match(Ag);spf.style.load(a,b?b[1]:"",void 0)}else Cg(a)}
function Cg(a){var b=Dg(a),c=document.getElementById(b),d=c&&pe(c,"loaded");d||c&&!d||(c=Eg(a,b,function(){pe(c,"loaded")||(qe(c),ze(b),W(w(ye,b),0))}))}
function Eg(a,b,c){var d=document.createElement("link");d.id=b;d.onload=function(){c&&setTimeout(c,0)};
a=Mb(a);d.rel="stylesheet";d.href=a instanceof xb&&a.constructor===xb&&a.f===yb?a.b:"type_error:TrustedResourceUrl";(document.getElementsByTagName("head")[0]||document.body).appendChild(d);return d}
function Dg(a){var b=document.createElement("a");Hb(b,a);a=b.href.replace(/^[a-zA-Z]+:\/\//,"//");return"css-"+ua(a)}
;var Fg={"api.invalidparam":2,auth:150,"drm.auth":150,"heartbeat.net":150,"heartbeat.servererror":150,"heartbeat.stop":150,"html5.unsupportedads":5,"fmt.noneavailable":5,"fmt.decode":5,"fmt.unplayable":5,"html5.missingapi":5,"html5.unsupportedlive":5,"drm.unavailable":5};var Gg;var Hg=F,Hg=Hg.toLowerCase();if(-1!=Hg.indexOf("android")){var Ig=Hg.match(/android\D*(\d\.\d)[^\;|\)]*[\;\)]/);if(Ig)Gg=Number(Ig[1]);else{var Jg={cupcake:1.5,donut:1.6,eclair:2,froyo:2.2,gingerbread:2.3,honeycomb:3,"ice cream sandwich":4,jellybean:4.1,kitkat:4.4,lollipop:5.1,marshmallow:6,nougat:7.1},Kg=[],Lg=0,Mg;for(Mg in Jg)Kg[Lg++]=Mg;var Ng=Hg.match("("+Kg.join("|")+")");Gg=Ng?Jg[Ng[0]]:0}}else Gg=void 0;var Og=['video/mp4; codecs="avc1.42001E, mp4a.40.2"','video/webm; codecs="vp8.0, vorbis"'],Pg=['audio/mp4; codecs="mp4a.40.2"'];var Qg;var Rg=F,Sg=Rg.match(/\((iPad|iPhone|iPod)( Simulator)?;/);if(!Sg||2>Sg.length)Qg=void 0;else{var Tg=Rg.match(/\((iPad|iPhone|iPod)( Simulator)?; (U; )?CPU (iPhone )?OS (\d+_\d)[_ ]/);Qg=Tg&&6==Tg.length?Number(Tg[5].replace("_",".")):0}0<=Qg&&0<=F.search("Safari")&&F.search("Version");function Ug(a){P.call(this);this.b=[];this.g=a||this}
z(Ug,P);function Vg(a,b,c,d){d=re(v(d,a.g));d={target:b,name:c,ra:d};b.addEventListener(c,d.ra,void 0);a.b.push(d)}
function Wg(a){for(;a.b.length;){var b=a.b.pop();b.target.removeEventListener(b.name,b.ra)}}
Ug.prototype.o=function(){Wg(this);Ug.A.o.call(this)};var Xg=r("ytLoggingLatencyUsageStats_")||{};q("ytLoggingLatencyUsageStats_",Xg,void 0);var Yg=0;function Zg(a){Xg[a]=Xg[a]||{count:0};var b=Xg[a];b.count++;b.time=je();Yg||(Yg=of($g,0));return 10<b.count?(11==b.count&&jf(Error("CSI data exceeded logging limit with key: "+a)),!0):!1}
function $g(){var a=je(),b;for(b in Xg)6E4<a-Xg[b].time&&delete Xg[b];Yg=0}
;function ah(a,b){this.version=a;this.args=b}
;function bh(a){this.topic=a}
bh.prototype.toString=function(){return this.topic};var ch=r("ytPubsub2Pubsub2Instance")||new R;R.prototype.subscribe=R.prototype.subscribe;R.prototype.unsubscribeByKey=R.prototype.K;R.prototype.publish=R.prototype.V;R.prototype.clear=R.prototype.clear;q("ytPubsub2Pubsub2Instance",ch,void 0);var dh=r("ytPubsub2Pubsub2SubscribedKeys")||{};q("ytPubsub2Pubsub2SubscribedKeys",dh,void 0);var eh=r("ytPubsub2Pubsub2TopicToKeys")||{};q("ytPubsub2Pubsub2TopicToKeys",eh,void 0);var fh=r("ytPubsub2Pubsub2IsAsync")||{};q("ytPubsub2Pubsub2IsAsync",fh,void 0);
q("ytPubsub2Pubsub2SkipSubKey",null,void 0);function gh(a){var b=hh,c=r("ytPubsub2Pubsub2Instance");c&&c.publish.call(c,b.toString(),b,a)}
;var X=window.performance||window.mozPerformance||window.msPerformance||window.webkitPerformance||{};function ih(a,b){ah.call(this,1,arguments)}
z(ih,ah);var hh=new bh("timing-sent");var jh=x().toString();var kh={vc:!0},lh={ad_at:"adType",cpn:"clientPlaybackNonce",csn:"clientScreenNonce",is_nav:"isNavigation",yt_lt:"loadType",yt_ad:"isMonetized",yt_ad_pr:"prerollAllowed",yt_red:"isRedSubscriber",yt_vis:"isVisible",docid:"videoId",plid:"videoId",fmt:"playerInfo.itag"},mh="ap c cver ei yt_fss yt_li".split(" "),nh=["isNavigation","isMonetized","prerollAllowed","isRedSubscriber","isVisible"];
function oh(a){if("_"!=a[0]){var b=a;X.mark&&(0==b.lastIndexOf("mark_",0)||(b="mark_"+b),X.mark(b))}var b=ph(),c=je();b[a]&&(b["_"+a]=b["_"+a]||[b[a]],b["_"+a].push(c));b[a]=c;qh()["tick_"+a]=void 0;je();U("csi_on_gel")?(b=rh(),"_start"==a?Zg("baseline_"+b)||ig("latencyActionBaselined",{clientActionNonce:b},qg,void 0):Zg("tick_"+a+"_"+b)||ig("latencyActionTicked",{tickName:a,clientActionNonce:b},qg,void 0),a=!0):a=!1;if(a=!a)a=!r("yt.timing.pingSent_");if(a&&(b=T("TIMING_ACTION",void 0),a=ph(),r("ytglobal.timingready_")&&
b&&a._start&&sh())){b=!0;c=T("TIMING_WAIT",[]);if(c.length)for(var d=0,e=c.length;d<e;++d)if(!(c[d]in a)){b=!1;break}b&&th()}}
function uh(){var a=vh().info.yt_lt="hot_bg";qh().info_yt_lt=a;if(U("csi_on_gel"))if("yt_lt"in lh){var b={},c=lh.yt_lt.split(".");0<=va(nh,c)&&(a=!!a);for(var d=b,e=0;e<c.length-1;e++)d[c[e]]=d[c[e]]||{},d=d[c[e]];b[c[c.length-1]]=a;a=rh();c=Object.keys(b).join("");Zg("info_"+c+"_"+a)||(b.clientActionNonce=a,ig("latencyActionInfo",b,qg))}else 0<=va(mh,"yt_lt")||V(Error("Unknown label yt_lt logged with GEL CSI."))}
function sh(){var a=ph();if(a.aft)return a.aft;for(var b=T("TIMING_AFT_KEYS",["ol"]),c=b.length,d=0;d<c;d++){var e=a[b[d]];if(e)return e}return NaN}
function th(){if(!U("csi_on_gel")){var a=ph(),b=vh().info,c=a._start,d;for(d in a)if(0==d.lastIndexOf("_",0)&&ia(a[d])){var e=d.slice(1);if(e in kh){var f=wa(a[d],function(a){return Math.round(a-c)});
b["all_"+e]=f.join()}delete a[d]}e=!!b.ap;if(f=r("ytglobal.timingReportbuilder_")){if(f=f(a,b,void 0))wh(f,e),xh(),yh(),zh(!1,void 0),T("TIMING_ACTION")&&S("PREVIOUS_ACTION",T("TIMING_ACTION")),S("TIMING_ACTION","")}else{var g=T("CSI_SERVICE_NAME","youtube"),f={v:2,s:g,action:T("TIMING_ACTION",void 0)},h=b.srt;void 0!==a.srt&&delete b.srt;if(b.h5jse){var k=window.location.protocol+r("ytplayer.config.assets.js");(k=X.getEntriesByName?X.getEntriesByName(k)[0]:null)?b.h5jse=Math.round(b.h5jse-k.responseEnd):
delete b.h5jse}a.aft=sh();Ah()&&"youtube"==g&&(uh(),g=a.vc,k=a.pbs,delete a.aft,b.aft=Math.round(k-g));for(var m in b)"_"!=m.charAt(0)&&(f[m]=b[m]);a.ps=je();b={};m=[];for(d in a)"_"!=d.charAt(0)&&(g=Math.round(a[d]-c),b[d]=g,m.push(d+"."+g));f.rt=m.join(",");(a=r("ytdebug.logTiming"))&&a(f,b);wh(f,e,void 0);gh(new ih(b.aft+(h||0),void 0))}}}
var yh=v(X.clearResourceTimings||X.webkitClearResourceTimings||X.mozClearResourceTimings||X.msClearResourceTimings||X.oClearResourceTimings||t,X);
function wh(a,b,c){if(U("debug_csi_data")){var d=r("yt.timing.csiData");d||(d=[],q("yt.timing.csiData",d,void 0));d.push({page:location.href,time:new Date,args:a})}var d="",e;for(e in a)d+="&"+e+"="+a[e];a="/csi_204?"+d.substring(1);if(window.navigator&&window.navigator.sendBeacon&&b)try{window.navigator&&window.navigator.sendBeacon&&window.navigator.sendBeacon(a,"")||gf(a,void 0)}catch(f){gf(a,void 0)}else gf(a);zh(!0,c)}
function rh(){var a=vh().nonce;if(!a){var b;a:{if(window.crypto&&window.crypto.getRandomValues)try{var c=Array(16),d=new Uint8Array(16);window.crypto.getRandomValues(d);for(a=0;a<c.length;a++)c[a]=d[a];b=c;break a}catch(e){}b=Array(16);for(c=0;16>c;c++){d=x();for(a=0;a<d%23;a++)b[c]=Math.random();b[c]=Math.floor(256*Math.random())}if(jh)for(c=1,d=0;d<jh.length;d++)b[c%16]=b[c%16]^b[(c-1)%16]/4^jh.charCodeAt(d),c++}c=[];for(d=0;d<b.length;d++)c.push("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_".charAt(b[d]&
63));a=c.join("");vh().nonce=a}return a}
function ph(){return vh().tick}
function qh(){var a=vh();"gel"in a||(a.gel={});return a.gel}
function vh(){return r("ytcsi.data_")||xh()}
function xh(){var a={tick:{},info:{}};q("ytcsi.data_",a,void 0);return a}
function zh(a,b){q("yt.timing."+(b||"")+"pingSent_",a,void 0)}
function Ah(){var a=ph(),b=a.pbr,c=a.vc,a=a.pbs;return b&&c&&a&&b<c&&c<a&&1==vh().info.yt_pvis}
;function Bh(a,b){P.call(this);this.w=this.l=a;this.T=b;this.C=!1;this.g={};this.Z=this.S=null;this.U=new R;td(this,w(ud,this.U));this.j={};this.L=this.aa=this.i=this.ha=this.b=null;this.W=!1;this.M=this.B=this.H=this.R=null;this.ba={};this.Ca=["onReady"];this.X=new Ug(this);td(this,w(ud,this.X));this.fa=null;this.oa=NaN;this.Y={};Ch(this);this.G("onDetailedError",v(this.Ta,this));this.G("onTabOrderChange",v(this.Ea,this));this.G("onTabAnnounce",v(this.pa,this));this.G("WATCH_LATER_VIDEO_ADDED",v(this.Ua,
this));this.G("WATCH_LATER_VIDEO_REMOVED",v(this.Va,this));qb||(this.G("onMouseWheelCapture",v(this.Ra,this)),this.G("onMouseWheelRelease",v(this.Sa,this)));this.G("onAdAnnounce",v(this.pa,this));this.N=new Ug(this);td(this,w(ud,this.N));this.ga=!1;this.ea=null}
z(Bh,P);var Dh=["drm.unavailable","fmt.noneavailable","html5.missingapi","html5.unsupportedads","html5.unsupportedlive"];l=Bh.prototype;l.na=function(a,b){this.f||(Eh(this,a),Fh(this,b),this.C&&Gh(this))};
function Eh(a,b){a.ha=b;a.b=xg(b);a.i=a.b.attrs.id||a.i;"video-player"==a.i&&(a.i=a.T,a.b.attrs.id=a.T);a.w.id==a.i&&(a.i+="-player",a.b.attrs.id=a.i);a.b.args.enablejsapi="1";a.b.args.playerapiid=a.T;a.aa||(a.aa=Hh(a,a.b.args.jsapicallback||"onYouTubePlayerReady"));a.b.args.jsapicallback=null;var c=a.b.attrs.width;c&&(a.w.style.width=Ob(Number(c)||c));if(c=a.b.attrs.height)a.w.style.height=Ob(Number(c)||c)}
l.Ha=function(){return this.ha};
function Gh(a){a.b.loaded||(a.b.loaded=!0,"0"!=a.b.args.autoplay?a.g.loadVideoByPlayerVars(a.b.args):a.g.cueVideoByPlayerVars(a.b.args))}
function Ih(a){if(!p(a.b.disable.flash)){var b=a.b.disable,c;c=zg(yg.getInstance(),a.b.minVersion);b.flash=!c}return!a.b.disable.flash}
function Jh(a,b){if((!b||(5!=(Fg[b.errorCode]||5)?0:-1!=Dh.indexOf(b.errorCode)))&&Ih(a)){var c=Kh(a);c&&c.stopVideo&&c.stopVideo();var d=a.b;c&&c.getUpdatedConfigurationData&&(c=c.getUpdatedConfigurationData(),d=wg(c));d.args.autoplay=1;d.args.html5_unavailable="1";Eh(a,d);Fh(a,"flash")}}
function Fh(a,b){if(!a.f){if(!b){var c;if(!(c=!a.b.html5&&Ih(a))){if(!p(a.b.disable.html5)){var d;c=!0;void 0!=a.b.args.deviceHasDisplay&&(c=a.b.args.deviceHasDisplay);if(2.2==Gg)d=!0;else{a:{var e=c;c=r("yt.player.utils.videoElement_");c||(c=document.createElement("VIDEO"),q("yt.player.utils.videoElement_",c,void 0));try{if(c.canPlayType)for(var e=e?Og:Pg,f=0;f<e.length;f++)if(c.canPlayType(e[f])){d=null;break a}d="fmt.noneavailable"}catch(g){d="html5.missingapi"}}d=!d}d&&(d=Lh(a)||a.b.assets.js);
a.b.disable.html5=!d;d||(a.b.args.html5_unavailable="1")}c=!!a.b.disable.html5}b=c?Ih(a)?"flash":"unsupported":"html5"}("flash"==b?a.lb:a.mb).call(a)}}
function Lh(a){var b=!0,c=Kh(a);c&&a.b&&(a=a.b,b=pe(c,"version")==a.assets.js);return b&&!!r("yt.player.Application.create")}
l.mb=function(){if(!this.W){var a=Lh(this);if(a&&"html5"==Mh(this))this.L="html5",this.C||this.P();else if(Nh(this),this.L="html5",a&&this.H)this.l.appendChild(this.H),this.P();else{this.b.loaded=!0;var b=!1;this.R=v(function(){b=!0;var a=this.l,d=xg(this.b);r("yt.player.Application.create")(a,d);this.P()},this);
this.W=!0;a?this.R():(Je(this.b.assets.js,this.R),Bg(this.b.assets.css),Oh(this)&&!b&&q("yt.player.Application.create",null,void 0))}}};
l.lb=function(){var a=xg(this.b);if(!this.B){var b=Kh(this);b&&(this.B=document.createElement("SPAN"),this.B.tabIndex=0,Vg(this.X,this.B,"focus",this.ta),this.M=document.createElement("SPAN"),this.M.tabIndex=0,Vg(this.X,this.M,"focus",this.ta),b.parentNode&&b.parentNode.insertBefore(this.B,b),b.parentNode&&b.parentNode.insertBefore(this.M,b.nextSibling))}a.attrs.width=a.attrs.width||"100%";a.attrs.height=a.attrs.height||"100%";if("flash"==Mh(this))this.L="flash",this.C||this.P();else{Nh(this);this.L=
"flash";this.b.loaded=!0;var b=yg.getInstance(),c=(-1<b.i.indexOf("Gnash")&&-1==b.i.indexOf("AVM2")||9==b.b&&1==b.f||9==b.b&&0==b.f&&1==b.g?0:9<=b.b)||-1<navigator.userAgent.indexOf("Sony/COM2")&&!zg(b,9,1,58)?a.url:a.urlV9As2;window!=window.top&&document.referrer&&(a.args.framer=document.referrer.substring(0,128));b=this.l;if(c){var b=u(b)?Ib(b):b,d=Ha(a.attrs);d.tabindex=0;var e=Ha(a.params);e.flashvars=Ue(a.args);if(ke){d.classid="clsid:D27CDB6E-AE6D-11cf-96B8-444553540000";e.movie=c;var a=document.createElement("object"),
f;for(f in d)a.setAttribute(f,d[f]);for(var g in e)f=document.createElement("param"),f.setAttribute("name",g),f.setAttribute("value",e[g]),a.appendChild(f)}else{d.type="application/x-shockwave-flash";d.src=c;a=document.createElement("embed");a.setAttribute("name",d.id);for(var h in d)a.setAttribute(h,d[h]);for(var k in e)a.setAttribute(k,e[k])}g=document.createElement("div");g.appendChild(a);b.innerHTML=g.innerHTML}this.P()}};
l.ta=function(){Kh(this).focus()};
function Kh(a){var b=Ib(a.i);!b&&a.w&&a.w.querySelector&&(b=a.w.querySelector("#"+a.i));return b}
l.P=function(){if(!this.f){var a=Kh(this),b=!1;try{a&&a.getApiInterface&&a.getApiInterface()&&(b=!0)}catch(f){}if(b)if(this.W=!1,a.isNotServable&&a.isNotServable(this.b.args.video_id))Jh(this);else{Ch(this);this.C=!0;a=Kh(this);a.addEventListener&&(this.S=Ph(this,a,"addEventListener"));a.removeEventListener&&(this.Z=Ph(this,a,"removeEventListener"));for(var b=a.getApiInterface(),b=b.concat(a.getInternalApiInterface()),c=0;c<b.length;c++){var d=b[c];this.g[d]||(this.g[d]=Ph(this,a,d))}for(var e in this.j)this.S(e,
this.j[e]);Gh(this);this.aa&&this.aa(this.g);this.U.V("onReady",this.g)}else this.oa=W(v(this.P,this),50)}};
function Ph(a,b,c){var d=b[c];return function(){try{return a.fa=null,d.apply(b,arguments)}catch(e){"Bad NPObject as private data!"!=e.message&&"sendAbandonmentPing"!=c&&(e.message+=" ("+c+")",a.fa=e,V(e,"WARNING"))}}}
function Ch(a){a.C=!1;if(a.Z)for(var b in a.j)a.Z(b,a.j[b]);for(var c in a.Y)window.clearTimeout(parseInt(c,10));a.Y={};a.S=null;a.Z=null;for(var d in a.g)a.g[d]=null;a.g.addEventListener=v(a.G,a);a.g.removeEventListener=v(a.bb,a);a.g.destroy=v(a.dispose,a);a.g.getLastError=v(a.Ia,a);a.g.getPlayerType=v(a.Ja,a);a.g.getCurrentVideoConfig=v(a.Ha,a);a.g.loadNewVideoConfig=v(a.na,a);a.g.isReady=v(a.nb,a)}
l.nb=function(){return this.C};
l.G=function(a,b){if(!this.f){var c=Hh(this,b);if(c){if(!(0<=va(this.Ca,a)||this.j[a])){var d=Qh(this,a);this.S&&this.S(a,d)}this.U.subscribe(a,c);"onReady"==a&&this.C&&W(w(c,this.g),0)}}};
l.bb=function(a,b){if(!this.f){var c=Hh(this,b);c&&Xd(this.U,a,c)}};
function Hh(a,b){var c=b;if("string"==typeof b){if(a.ba[b])return a.ba[b];c=function(){var a=r(b);a&&a.apply(n,arguments)};
a.ba[b]=c}return c?c:null}
function Qh(a,b){var c="ytPlayer"+b+a.T;a.j[b]=c;n[c]=function(c){var d=W(function(){if(!a.f){a.U.V(b,c);var e=a.Y,g=String(d);g in e&&delete e[g]}},0);
Ga(a.Y,String(d))};
return c}
l.Ea=function(a){a=a?Kb:Jb;for(var b=a(document.activeElement);b&&(1!=b.nodeType||b==this.B||b==this.M||(b.focus(),b!=document.activeElement));)b=a(b)};
l.pa=function(a){ze("a11y-announce",a)};
l.Ta=function(a){Jh(this,a)};
l.Ua=function(a){ze("WATCH_LATER_VIDEO_ADDED",a)};
l.Va=function(a){ze("WATCH_LATER_VIDEO_REMOVED",a)};
l.Ra=function(){if(!this.ga){if(ub){var a=document,b=a.scrollingElement?a.scrollingElement:bb||"CSS1Compat"!=a.compatMode?a.body||a.documentElement:a.documentElement,a=a.parentWindow||a.defaultView;this.ea=I&&J("10")&&a.pageYOffset!=b.scrollTop?new Ba(b.scrollLeft,b.scrollTop):new Ba(a.pageXOffset||b.scrollLeft,a.pageYOffset||b.scrollTop);Vg(this.N,window,"scroll",this.Ya);Vg(this.N,this.l,"touchmove",this.Xa)}else Vg(this.N,this.l,"mousewheel",this.ua),Vg(this.N,this.l,"wheel",this.ua);this.ga=!0}};
l.Sa=function(){Wg(this.N);this.ga=!1};
l.ua=function(a){a=a||window.event;a.returnValue=!1;a.preventDefault&&a.preventDefault()};
l.Ya=function(){window.scrollTo(this.ea.b,this.ea.f)};
l.Xa=function(a){a.preventDefault()};
l.Ja=function(){return this.L||Mh(this)};
l.Ia=function(){return this.fa};
function Mh(a){return(a=Kh(a))?"div"==a.tagName.toLowerCase()?"html5":"flash":null}
function Nh(a){oh("dcp");a.cancel();Ch(a);a.L=null;a.b&&(a.b.loaded=!1);var b=Kh(a);"html5"==Mh(a)?Lh(a)||!Oh(a)?a.H=b:(b&&b.destroy&&b.destroy(),a.H=null):b&&b.destroy&&b.destroy();for(var b=a.l,c;c=b.firstChild;)b.removeChild(c);Wg(a.X);a.B=null;a.M=null}
l.cancel=function(){this.R&&Ie(this.b.assets.js,this.R);window.clearTimeout(this.oa);this.W=!1};
l.o=function(){Nh(this);if(this.H&&this.b)try{this.H.destroy()}catch(b){V(b)}this.ba=null;for(var a in this.j)n[this.j[a]]=null;this.ha=this.b=this.g=null;delete this.l;delete this.w;Bh.A.o.call(this)};
function Oh(a){return a.b&&a.b.args&&a.b.args.fflags?-1!=a.b.args.fflags.indexOf("player_destroy_old_version=true"):!1}
;var Rh={},Sh="player_uid_"+(1E9*Math.random()>>>0);function Th(a){var b="player",b=u(b)?Ib(b):b;a=wg(a);var c=Sh+"_"+(b[ma]||(b[ma]=++na)),d=Rh[c];if(d)return d.na(a),d.g;d=new Bh(b,c);Rh[c]=d;ze("player-added",d.g);td(d,w(Uh,d));W(function(){d.na(a)},0);
return d.g}
function Uh(a){Rh[a.T]=null}
;function Vh(a){return(0==a.search("cue")||0==a.search("load"))&&"loadModule"!=a}
function Wh(a,b,c){u(a)&&(a={mediaContentUrl:a,startSeconds:b,suggestedQuality:c});b=/\/([ve]|embed)\/([^#?]+)/.exec(a.mediaContentUrl);a.videoId=b&&b[2]?b[2]:null;return Xh(a)}
function Xh(a,b,c){if(la(a)){b="endSeconds startSeconds mediaContentUrl suggestedQuality videoId two_stage_token".split(" ");c={};for(var d=0;d<b.length;d++){var e=b[d];a[e]&&(c[e]=a[e])}return c}return{videoId:a,startSeconds:b,suggestedQuality:c}}
function Yh(a,b,c,d){if(la(a)&&!ia(a)){b="playlist list listType index startSeconds suggestedQuality".split(" ");c={};for(d=0;d<b.length;d++){var e=b[d];a[e]&&(c[e]=a[e])}return c}c={index:b,startSeconds:c,suggestedQuality:d};u(a)&&16==a.length?c.list="PL"+a:c.playlist=a;return c}
function Zh(a){var b=a.video_id||a.videoId;if(u(b)){var c=Uf()||{},d=Uf()||{};p(void 0)?d[b]=void 0:delete d[b];var e=x()+3E5,f=Wf;if(f&&window.JSON){u(d)||(d=JSON.stringify(d,void 0));try{f.set("yt-player-two-stage-token",d,e)}catch(g){f.remove("yt-player-two-stage-token")}}(b=c[b])&&(a.two_stage_token=b)}}
;function $h(a){P.call(this);this.g=a;this.g.subscribe("command",this.xa,this);this.i={};this.j=!1}
z($h,P);l=$h.prototype;l.start=function(){this.j||this.f||(this.j=!0,ai(this.g,"RECEIVING"))};
l.xa=function(a,b){if(this.j&&!this.f){var c=b||{};switch(a){case "addEventListener":if(u(c.event)&&(c=c.event,!(c in this.i))){var d=v(this.eb,this,c);this.i[c]=d;this.addEventListener(c,d)}break;case "removeEventListener":u(c.event)&&bi(this,c.event);break;default:this.b.isReady()&&this.b[a]&&(c=ci(a,b||{}),c=this.b[a].apply(this.b,c),(c=di(a,c))&&this.j&&!this.f&&ai(this.g,a,c))}}};
l.eb=function(a,b){this.j&&!this.f&&ai(this.g,a,this.ia(a,b))};
l.ia=function(a,b){if(null!=b)return{value:b}};
function bi(a,b){b in a.i&&(a.removeEventListener(b,a.i[b]),delete a.i[b])}
l.o=function(){var a=this.g;a.f||Xd(a.b,"command",this.xa,this);this.g=null;for(var b in this.i)bi(this,b);$h.A.o.call(this)};function ei(a,b){$h.call(this,b);this.b=a;this.start()}
z(ei,$h);ei.prototype.addEventListener=function(a,b){this.b.addEventListener(a,b)};
ei.prototype.removeEventListener=function(a,b){this.b.removeEventListener(a,b)};
function ci(a,b){switch(a){case "loadVideoById":return b=Xh(b),Zh(b),[b];case "cueVideoById":return b=Xh(b),Zh(b),[b];case "loadVideoByPlayerVars":return Zh(b),[b];case "cueVideoByPlayerVars":return Zh(b),[b];case "loadPlaylist":return b=Yh(b),Zh(b),[b];case "cuePlaylist":return b=Yh(b),Zh(b),[b];case "seekTo":return[b.seconds,b.allowSeekAhead];case "playVideoAt":return[b.index];case "setVolume":return[b.volume];case "setPlaybackQuality":return[b.suggestedQuality];case "setPlaybackRate":return[b.suggestedRate];
case "setLoop":return[b.loopPlaylists];case "setShuffle":return[b.shufflePlaylist];case "getOptions":return[b.module];case "getOption":return[b.module,b.option];case "setOption":return[b.module,b.option,b.value];case "handleGlobalKeyDown":return[b.keyCode,b.shiftKey]}return[]}
function di(a,b){switch(a){case "isMuted":return{muted:b};case "getVolume":return{volume:b};case "getPlaybackRate":return{playbackRate:b};case "getAvailablePlaybackRates":return{availablePlaybackRates:b};case "getVideoLoadedFraction":return{videoLoadedFraction:b};case "getPlayerState":return{playerState:b};case "getCurrentTime":return{currentTime:b};case "getPlaybackQuality":return{playbackQuality:b};case "getAvailableQualityLevels":return{availableQualityLevels:b};case "getDuration":return{duration:b};
case "getVideoUrl":return{videoUrl:b};case "getVideoEmbedCode":return{videoEmbedCode:b};case "getPlaylist":return{playlist:b};case "getPlaylistIndex":return{playlistIndex:b};case "getOptions":return{options:b};case "getOption":return{option:b}}}
ei.prototype.ia=function(a,b){switch(a){case "onReady":return;case "onStateChange":return{playerState:b};case "onPlaybackQualityChange":return{playbackQuality:b};case "onPlaybackRateChange":return{playbackRate:b};case "onError":return{errorCode:b}}return ei.A.ia.call(this,a,b)};
ei.prototype.o=function(){ei.A.o.call(this);delete this.b};function fi(a,b,c,d){P.call(this);this.g=b||null;this.B="*";this.i=c||null;this.sessionId=null;this.channel=d||null;this.H=!!a;this.w=v(this.C,this);window.addEventListener("message",this.w)}
aa(fi,P);
fi.prototype.C=function(a){if(!("*"!=this.i&&a.origin!=this.i||this.g&&a.source!=this.g)&&u(a.data)){var b;try{b=xd(a.data)}catch(c){return}if(!(null==b||this.H&&(this.sessionId&&this.sessionId!=b.id||this.channel&&this.channel!=b.channel))&&b)switch(b.event){case "listening":"null"!=a.origin?this.i=this.B=a.origin:V(Error("MessageEvent origin is null"),"WARNING");this.g=a.source;this.sessionId=b.id;this.b&&(this.b(),this.b=null);break;case "command":this.j&&(this.l&&!(0<=va(this.l,b.func))||this.j(b.func,
b.args))}}};
fi.prototype.sendMessage=function(a,b){var c=b||this.g;if(c){this.sessionId&&(a.id=this.sessionId);this.channel&&(a.channel=this.channel);try{var d=zd(a);c.postMessage(d,this.B)}catch(e){V(e,"WARNING")}}};
fi.prototype.o=function(){window.removeEventListener("message",this.w);P.prototype.o.call(this)};function gi(a,b,c){fi.call(this,a,b,c||T("POST_MESSAGE_ORIGIN",void 0)||window.document.location.protocol+"//"+window.document.location.hostname,"widget");this.l=this.b=this.j=null}
aa(gi,fi);function hi(){var a=!!T("WIDGET_ID_ENFORCE"),a=this.f=new gi(a),b=v(this.ab,this);a.j=b;a.l=null;this.f.channel="widget";if(a=T("WIDGET_ID"))this.f.sessionId=a;this.i=[];this.l=!1;this.j={}}
l=hi.prototype;l.ab=function(a,b){if("addEventListener"==a&&b){var c=b[0];this.j[c]||"onReady"==c||(this.addEventListener(c,ii(this,c)),this.j[c]=!0)}else this.Aa(a,b)};
l.Aa=function(){};
function ii(a,b){return v(function(a){this.sendMessage(b,a)},a)}
l.addEventListener=function(){};
l.Ga=function(){this.l=!0;this.sendMessage("initialDelivery",this.ja());this.sendMessage("onReady");C(this.i,this.za,this);this.i=[]};
l.ja=function(){return null};
function ji(a,b){a.sendMessage("infoDelivery",b)}
l.za=function(a){this.l?this.f.sendMessage(a):this.i.push(a)};
l.sendMessage=function(a,b){this.za({event:a,info:void 0==b?null:b})};
l.dispose=function(){this.f=null};function ki(a){hi.call(this);this.b=a;this.g=[];this.addEventListener("onReady",v(this.Wa,this));this.addEventListener("onVideoProgress",v(this.ib,this));this.addEventListener("onVolumeChange",v(this.jb,this));this.addEventListener("onApiChange",v(this.cb,this));this.addEventListener("onPlaybackQualityChange",v(this.fb,this));this.addEventListener("onPlaybackRateChange",v(this.gb,this));this.addEventListener("onStateChange",v(this.hb,this))}
z(ki,hi);l=ki.prototype;l.Aa=function(a,b){if(this.b[a]){b=b||[];if(0<b.length&&Vh(a)){var c;c=b;if(la(c[0])&&!ia(c[0]))c=c[0];else{var d={};switch(a){case "loadVideoById":case "cueVideoById":d=Xh.apply(window,c);break;case "loadVideoByUrl":case "cueVideoByUrl":d=Wh.apply(window,c);break;case "loadPlaylist":case "cuePlaylist":d=Yh.apply(window,c)}c=d}Zh(c);b.length=1;b[0]=c}this.b[a].apply(this.b,b);Vh(a)&&ji(this,this.ja())}};
l.Wa=function(){var a=v(this.Ga,this);this.f.b=a};
l.addEventListener=function(a,b){this.g.push({eventType:a,listener:b});this.b.addEventListener(a,b)};
l.ja=function(){if(!this.b)return null;var a=this.b.getApiInterface();ya(a,"getVideoData");for(var b={apiInterface:a},c=0,d=a.length;c<d;c++){var e=a[c],f=e;if(0==f.search("get")||0==f.search("is")){var f=e,g=0;0==f.search("get")?g=3:0==f.search("is")&&(g=2);f=f.charAt(g).toLowerCase()+f.substr(g+1);try{var h=this.b[e]();b[f]=h}catch(k){}}}b.videoData=this.b.getVideoData();b.currentTimeLastUpdated_=x()/1E3;return b};
l.hb=function(a){a={playerState:a,currentTime:this.b.getCurrentTime(),duration:this.b.getDuration(),videoData:this.b.getVideoData(),videoStartBytes:0,videoBytesTotal:this.b.getVideoBytesTotal(),videoLoadedFraction:this.b.getVideoLoadedFraction(),playbackQuality:this.b.getPlaybackQuality(),availableQualityLevels:this.b.getAvailableQualityLevels(),videoUrl:this.b.getVideoUrl(),playlist:this.b.getPlaylist(),playlistIndex:this.b.getPlaylistIndex(),currentTimeLastUpdated_:x()/1E3,playbackRate:this.b.getPlaybackRate(),
mediaReferenceTime:this.b.getMediaReferenceTime()};this.b.getProgressState&&(a.progressState=this.b.getProgressState());this.b.getStoryboardFormat&&(a.storyboardFormat=this.b.getStoryboardFormat());ji(this,a)};
l.fb=function(a){ji(this,{playbackQuality:a})};
l.gb=function(a){ji(this,{playbackRate:a})};
l.cb=function(){for(var a=this.b.getOptions(),b={namespaces:a},c=0,d=a.length;c<d;c++){var e=a[c],f=this.b.getOptions(e);b[e]={options:f};for(var g=0,h=f.length;g<h;g++){var k=f[g],m=this.b.getOption(e,k);b[e][k]=m}}this.sendMessage("apiInfoDelivery",b)};
l.jb=function(){ji(this,{muted:this.b.isMuted(),volume:this.b.getVolume()})};
l.ib=function(a){a={currentTime:a,videoBytesLoaded:this.b.getVideoBytesLoaded(),videoLoadedFraction:this.b.getVideoLoadedFraction(),currentTimeLastUpdated_:x()/1E3,playbackRate:this.b.getPlaybackRate(),mediaReferenceTime:this.b.getMediaReferenceTime()};this.b.getProgressState&&(a.progressState=this.b.getProgressState());ji(this,a)};
l.dispose=function(){ki.A.dispose.call(this);for(var a=0;a<this.g.length;a++){var b=this.g[a];this.b.removeEventListener(b.eventType,b.listener)}this.g=[]};function li(){P.call(this);this.b=new R;td(this,w(ud,this.b))}
z(li,P);li.prototype.subscribe=function(a,b,c){return this.f?0:this.b.subscribe(a,b,c)};
li.prototype.K=function(a){return this.f?!1:this.b.K(a)};
li.prototype.l=function(a,b){this.f||this.b.V.apply(this.b,arguments)};function mi(a,b,c){li.call(this);this.g=a;this.i=b;this.j=c}
z(mi,li);function ai(a,b,c){if(!a.f){var d=a.g;d.f||a.i!=d.b||(a={id:a.j,command:b},c&&(a.data=c),d.b.postMessage(zd(a),d.i))}}
mi.prototype.o=function(){this.i=this.g=null;mi.A.o.call(this)};function ni(a,b,c){P.call(this);this.b=a;this.i=c;this.j=zf(window,"message",v(this.l,this));this.g=new mi(this,a,b);td(this,w(ud,this.g))}
z(ni,P);ni.prototype.l=function(a){var b;if(b=!this.f)if(b=a.origin==this.i)a:{b=this.b;do{var c;b:{c=a.source;do{if(c==b){c=!0;break b}if(c==c.parent)break;c=c.parent}while(null!=c);c=!1}if(c){b=!0;break a}b=b.opener}while(null!=b);b=!1}if(b&&(a=a.data,u(a))){try{a=xd(a)}catch(d){return}a.command&&(b=this.g,b.f||b.l("command",a.command,a.data))}};
ni.prototype.o=function(){Af(this.j);this.b=null;ni.A.o.call(this)};function oi(){var a=Ha(pi);return new Q(function(b,c){a.J=function(a){Re(a)?b(a):c(new qi("Request failed, status="+a.status,"net.badstatus"))};
a.onError=function(){c(new qi("Unknown request error","net.unknown"))};
a.O=function(){c(new qi("Request timed out","net.timeout"))};
Ye("//googleads.g.doubleclick.net/pagead/id",a)})}
function qi(a,b){B.call(this,a+", errorCode="+b);this.errorCode=b;this.name="PromiseAjaxError"}
aa(qi,B);function ri(a){B.call(this,a.message||a.description||a.name);this.Pa=a instanceof si;this.b=a instanceof Nd}
z(ri,B);ri.prototype.name="BiscottiError";function si(){B.call(this,"Biscotti ID is missing from server")}
z(si,B);si.prototype.name="BiscottiMissingError";function ti(a,b){this.f=a;this.b=b}
ti.prototype.then=function(a,b,c){try{if(p(this.f))return a?Kd(a.call(c,this.f)):Kd(this.f);if(p(this.b)){if(!b)return Ld(this.b);var d=b.call(c,this.b);return!p(d)&&this.b.b?Ld(this.b):Kd(d)}throw Error("Invalid Result_ state");}catch(e){return Ld(e)}};
Fd(ti);var pi={format:"RAW",method:"GET",timeout:5E3,withCredentials:!0},ui=null;function vi(){if(!ui){var a=v(wi,n,2),b=oi().then(xi);ui=Md(b,null,a,void 0)}return ui}
function xi(a){a=a.responseText;if(0!=a.lastIndexOf(")]}'",0))throw new si;a=JSON.parse(a.substr(4)).id;yi(a);ui=new ti(a);zi(18E5,2);return a}
function wi(a,b){var c=new ri(b);yi("");ui=new ti(void 0,c);0<a&&zi(12E4,a-1);throw c;}
function zi(a,b){W(function(){var a=v(wi,n,b),a=oi().then(xi,a);Md(a,null,t,void 0)},a)}
function yi(a){q("yt.ads.biscotti.lastId_",a,void 0)}
;function Ai(a){a.Pa&&Bi("")}
function Ci(a){!a||r("yt.ads.biscotti.getId_")||r("yt.www.ads.biscotti.getId_")||q("yt.ads.biscotti.getId_",vi,void 0);try{var b;try{var c=r("yt.ads.biscotti.getId_")||r("yt.www.ads.biscotti.getId_");b=c?c():vi()}catch(d){b=Ld(d)}b.then(Bi,Ai);W(Ci,18E5)}catch(d){V(d)}}
var Di=0;
function Bi(a){var b;a:{try{b=window.top.location.href}catch(G){b=2;break a}b=null!=b?b==window.document.location.href?0:1:2}b={dt:cc,flash:Ua||"0",frm:b};b.u_tz=-(new Date).getTimezoneOffset();var c;try{c=A.history.length}catch(G){c=0}b.u_his=c;b.u_java=!!A.navigator&&"unknown"!==typeof A.navigator.javaEnabled&&!!A.navigator.javaEnabled&&A.navigator.javaEnabled();A.screen&&(b.u_h=A.screen.height,b.u_w=A.screen.width,b.u_ah=A.screen.availHeight,b.u_aw=A.screen.availWidth,b.u_cd=A.screen.colorDepth);A.navigator&&
A.navigator.plugins&&(b.u_nplug=A.navigator.plugins.length);A.navigator&&A.navigator.mimeTypes&&(b.u_nmime=A.navigator.mimeTypes.length);b.bid=a;b.ca_type=Ta?"flash":"image";if(U("enable_server_side_search_pyv")||U("enable_server_side_mweb_search_pyv")){var d,e;a=window;try{e=a.screenX;var f=a.screenY}catch(G){}try{var g=a.outerWidth,h=a.outerHeight}catch(G){}try{var k=a.innerWidth,m=a.innerHeight}catch(G){}e=[a.screenLeft,a.screenTop,e,f,a.screen?a.screen.availWidth:void 0,a.screen?a.screen.availTop:
void 0,g,h,k,m];f=window.top||A;try{var y;if(f.document&&!f.document.body)y=new Ca(-1,-1);else{var E=(f||window).document,Z="CSS1Compat"==E.compatMode?E.documentElement:E.body;y=(new Ca(Z.clientWidth,Z.clientHeight)).round()}d=y}catch(G){d=new Ca(-12245933,-12245933)}y=0;window.SVGElement&&document.createElementNS&&(y|=1);d={bc:y,bih:d.height,biw:d.width,brdim:e.join(),vis:{visible:1,hidden:2,prerender:3,preview:4}[qa.webkitVisibilityState||qa.mozVisibilityState||qa.visibilityState||""]||0,wgl:!!A.WebGLRenderingContext};
for(var ta in d)b[ta]=d[ta]}b.bsq=Di++;df("//www.youtube.com/ad_data_204",{Ka:!1,D:b})}
;function Ei(){this.b=T("ALT_PREF_COOKIE_NAME","PREF");var a=tc.get(""+this.b,void 0);if(a)for(var a=unescape(a).split("&"),b=0;b<a.length;b++){var c=a[b].split("="),d=c[0];(c=c[1])&&(Y[d]=c.toString())}}
ea(Ei);var Y=r("yt.prefs.UserPrefs.prefs_")||{};q("yt.prefs.UserPrefs.prefs_",Y,void 0);function Fi(a){if(/^f([1-9][0-9]*)$/.test(a))throw Error("ExpectedRegexMatch: "+a);}
function Gi(a){if(!/^\w+$/.test(a))throw Error("ExpectedRegexMismatch: "+a);}
function Hi(a){a=void 0!==Y[a]?Y[a].toString():null;return null!=a&&/^[A-Fa-f0-9]+$/.test(a)?parseInt(a,16):null}
Ei.prototype.get=function(a,b){Gi(a);Fi(a);var c=void 0!==Y[a]?Y[a].toString():null;return null!=c?c:b?b:""};
Ei.prototype.set=function(a,b){Gi(a);Fi(a);if(null==b)throw Error("ExpectedNotNull");Y[a]=b.toString()};
Ei.prototype.remove=function(a){Gi(a);Fi(a);delete Y[a]};
Ei.prototype.clear=function(){Y={}};var Ii=null,Ji=null,Ki=null,Li={};function Mi(a){ig(a.payload_name,a.payload,U("enable_youtubei_innertube")?qg:mg,void 0,void 0)}
function Ni(a){var b=a.id;a=a.ve_type;var c=Ff++;a=new Ef(void 0,a,c,void 0);Li[b]=a;b=$f();c=Zf();b&&c&&Qf(rg(),b,c,a)}
function Oi(a){var b=a.csn;a=a.root_ve_type;if(b&&a&&(S("client-screen-nonce",b),S("ROOT_VE_TYPE",a),a=Zf()))for(var c in Li){var d=b,e=a,f=Li[c];Qf(rg(),d,e,f)}}
function Pi(a){Li[a.id]=new Ef(a.tracking_params)}
function Qi(a){var b=$f();a=Li[a.id];if(b&&a){var c=rg();Rf(c,{click:{csn:b,visualElement:Gf(a)}},void 0)}}
function Ri(a){a=a.ids;var b=$f();if(b){for(var c=[],d=0;d<a.length;d++){var e=Li[a[d]];e&&c.push(e)}c.length&&Sf(rg(),b,c)}}
function Si(){var a=Ii;a&&a.startInteractionLogging&&a.startInteractionLogging()}
;q("yt.setConfig",S,void 0);q("yt.config.set",S,void 0);q("yt.setMsg",nf,void 0);q("yt.msgs.set",nf,void 0);q("yt.logging.errors.log",jf,void 0);
q("writeEmbed",function(){var a=T("PLAYER_CONFIG",void 0);"1"!=a.privembed&&Ci(!0);"gvn"==a.args.ps&&(document.body.style.backgroundColor="transparent");var b=document.referrer,c=T("POST_MESSAGE_ORIGIN");window!=window.top&&b&&b!=document.URL&&(a.args.loaderUrl=b);T("LIGHTWEIGHT_AUTOPLAY")&&(a.args.autoplay="1");a.args.autoplay&&Zh(a.args);Ii=a=Th(a);a.addEventListener("onScreenChanged",Oi);a.addEventListener("onLogClientVeCreated",Ni);a.addEventListener("onLogServerVeCreated",Pi);a.addEventListener("onLogToGel",
Mi);a.addEventListener("onLogVeClicked",Qi);a.addEventListener("onLogVesShown",Ri);a.addEventListener("onReady",Si);b=T("POST_MESSAGE_ID","player");T("ENABLE_JS_API")?Ki=new ki(a):T("ENABLE_POST_API")&&u(b)&&u(c)&&(Ji=new ni(window.parent,b,c),Ki=new ei(a,Ji.g));T("BG_P")&&(T("BG_I")||T("BG_IU"))&&Le();rf()},void 0);
q("yt.www.watch.ads.restrictioncookie.spr",function(a){gf(a+"mac_204?action_fcts=1");return!0},void 0);
var Ti=re(function(){oh("ol");var a=Ei.getInstance(),b=1<window.devicePixelRatio;if(!!((Hi("f"+(Math.floor(119/31)+1))||0)&67108864)!=b){var c="f"+(Math.floor(119/31)+1),d=Hi(c)||0,d=b?d|67108864:d&-67108865;0==d?delete Y[c]:Y[c]=d.toString(16).toString();var a=a.b,b=[],e;for(e in Y)b.push(e+"="+escape(Y[e]));tc.set(""+a,b.join("&"),63072E3,"/","youtube.com")}}),Ui=re(function(){var a=Ii;
a&&a.sendAbandonmentPing&&a.sendAbandonmentPing();T("PL_ATT")&&(Ke=null);for(var a=0,b=pf.length;a<b;a++){var c=pf[a];if(!isNaN(c)){var d=r("yt.scheduler.instance.cancelJob");d?d(c):window.clearTimeout(c)}}pf.length=0;a=Fe("//static.doubleclick.net/instream/ad_status.js");if(b=document.getElementById(a))ye(a),b.parentNode.removeChild(b);qf=!1;S("DCLKSTAT",0);vd(Ki,Ji);if(a=Ii)a.removeEventListener("onScreenChanged",Oi),a.removeEventListener("onLogClientVeCreated",Ni),a.removeEventListener("onLogServerVeCreated",
Pi),a.removeEventListener("onLogToGel",Mi),a.removeEventListener("onLogVeClicked",Qi),a.removeEventListener("onLogVesShown",Ri),a.removeEventListener("onReady",Si),a.destroy();Li={}});
window.addEventListener?(window.addEventListener("load",Ti),window.addEventListener("unload",Ui)):window.attachEvent&&(window.attachEvent("onload",Ti),window.attachEvent("onunload",Ui));}).call(this);
