/**
 * 推广链接访问统计脚本
 * 使用示例（将下列脚本拷贝到页面底部）：
 * 
	<script type="text/javascript">
	var _graq = _graq || [];
	_graq.push(['_getlkid', 'gearBest']);
	
	(function() {
		var path = '';
	    var ga = document.createElement('script'); ga.type = 'text/javascript'; ga.async = true;
	    ga.src = ('https:' == document.location.protocol ? 'https://' : 'http://') + path + 'ga.js';
	    var s = document.getElementsByTagName('script')[0]; s.parentNode.insertBefore(ga, s);
	})();
	</script>
 */
(function() {
	/** 
	* 获取URL参数(类似PHP的$_GET)
	*
	* @param {string} name 参数
	* @param {string} str 待获取字符串
	*
	* @return {string} 参数值
	*/
	function _GRGET(name, str) {
		var pattern = new RegExp('[\?&]' + name + '=([^&]+)', 'g');
		var pattern2 = new RegExp('[\#&]' + name + '=([^&]+)', 'g');
		
		str = str || location.search;
		var arr, match = '';
	
		while ((arr = pattern.exec(str)) !== null || (arr = pattern2.exec(str)) !== null) {
			match = arr[1];
		}
	
		return match;
	}
	
	/**
	* 通过imgurl格式发送数据到服务器
	*
	* @param {object} 需要发送的数据对象
	*
	* @return {boolean} false
	*/
	function setData(params) {
		var args = '';
		for (var key in params) {
			if (args != '') {
				args += '&';
			}
			args += key + '=' + params[key];
		}
		
		var img = new Image(1, 1);
		img.src = "http://affiliate.gearbest.com/?" + args;
		
		return false;
	}
	
	/**
	* 统计各站点的推广链接
	*
	* @param {object}  全局对象params
	* @param {object}  站点id，网站补点必填信息
	* @return {boolean} false
	*/
	function getLkId(pa, webName) {
		// 拼接参数
		var newPa = {};
		var lkid = '';
        var pb_cid = '';
        var pb_refid = '';
        var postback_id = '';

		lkid = _GRGET("vip", pa.url);
		lkid = lkid && lkid.length ? lkid : _GRGET("lkid", pa.url);

		if (lkid && lkid.length) {
			newPa.url = encodeURIComponent(pa.url);
			newPa.web_id = webName;
			newPa.lkid = lkid;
			newPa.timestamp = (new Date()).getTime();
			newPa.reffer = encodeURIComponent(pa.referrer);
			
			setCookie('landingUrl', location.href, 30);
			setCookie('linkid', lkid, 30);    // ai表示 affiliate id
			setCookie('reffer_channel', newPa.reffer, 30);
			setCookie("utm_source", "xxxxxx", 1);  //去重,只保留最后的佣金平台 2016.06.23
			
			setData(newPa);
		}

        pb_cid   = _GRGET("cid", pa.url);
        pb_refid = _GRGET("refid", pa.url);

        postback_id = pb_cid && pb_cid.length ? pb_cid : pb_refid;
        if (postback_id && postback_id.length) {
            newPa.url = encodeURIComponent(pa.url);
            newPa.web_id = webName;
            newPa.postback_id = postback_id;
            newPa.timestamp = (new Date()).getTime();
            newPa.reffer = encodeURIComponent(pa.referrer);
            if (pb_cid && pb_cid.length) {
                postback_id = 'cid:' + pb_cid;
            } else {
                postback_id = 'refid:' + pb_refid;
            }
            setCookie('landingUrl', location.href, 30);
            setCookie('postback_id', postback_id, 30);    // ai表示 affiliate id
            setCookie('reffer_channel', newPa.reffer, 30);
            setCookie("utm_source", "xxxxxx", 1);  //去重,只保留最后的佣金平台 2016.06.23

            setData(newPa);
        }

		return false;
	}

	/**
	 * 设置cookie
	 */
	function setCookie(cname, cvalue, exdays) {
		var d = new Date();
		d.setTime(d.getTime() + (exdays*24*60*60*1000));
		var expires = "expires="+d.toUTCString();
		document.cookie = cname + "=" + cvalue + ";" + expires + ";path=/;domain=" + COOKIESDIAMON;
	}
	
	var params = {};
    //Document对象数据
    if(document) {
        params.domain = document.domain || ''; 
        params.url = document.URL || ''; 
        params.title = document.title || ''; 
        params.referrer = document.referrer || ''; 
    }   
    //Window对象数据
    if(window && window.screen) {
        params.sh = window.screen.height || 0;
        params.sw = window.screen.width || 0;
        params.cd = window.screen.colorDepth || 0;
    }   
    //navigator对象数据
    if(navigator) {
        params.lang = navigator.language || ''; 
    }
	
    //解析_graq配置
    if(_graq) {
        for(var i in _graq) {
            switch(_graq[i][0]) {
                case '_getlkid':
					getLkId(params, _graq[i][1]);

                    break;
                default:
                    break;
            }   
        }   
    }
})();