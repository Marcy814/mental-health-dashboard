/*! modernizr 3.6.0 (Custom Build) | MIT *
 * https://modernizr.com/download/?-cssanimations-cssgradients-csstransforms-csstransforms3d-csstransitions-flexbox-promises-svg-setclasses !*/
!function (e, n, A) {

    function o(e, n) {
        return typeof e === n
    }

    function s() {
        var e, n, A, s, a, i, l;
        for (var f in r)
            if (r.hasOwnProperty(f)) {
                if (e = [], n = r[f], n.name && (e.push(n.name.toLowerCase()), n.options && n.options.aliases && n.options.aliases.length))
                    for (A = 0; A < n.options.aliases.length; A++)
                        e.push(n.options.aliases[A].toLowerCase());
                for (s = o(n.fn, "function") ? n.fn() : n.fn, a = 0; a < e.length; a++)
                    i = e[a],
                    l = i.split("."),
                    1 === l.length ? Modernizr[l[0]] = s :
                    (!Modernizr[l[0]] || Modernizr[l[0]] instanceof Boolean || (Modernizr[l[0]] = new Boolean(Modernizr[l[0]])), Modernizr[l[0]][l[1]] = s),
                    t.push((s ? "" : "no-") + l.join("-"))
            }
    }

    function a(e) {
        var n = u.className,
            A = Modernizr._config.classPrefix || "";
        if (c && (n = n.baseVal), Modernizr._config.enableJSClass) {
            var o = new RegExp("(^|\\s)" + A + "no-js(\\s|$)");
            n = n.replace(o, "$1" + A + "js$2")
        }
        Modernizr._config.enableClasses && (n += " " + A + e.join(" " + A), c ? u.className.baseVal = n : u.className = n)
    }

    function i(e, n) {
        return a([e]), o(n, "function") && n(), Modernizr
    }

    var r = [],
        t = [],
        l = {
            _version: "3.6.0",
            _config: {
                classPrefix: "",
                enableClasses: !0,
                enableJSClass: !0,
                usePrefixes: !0
            },
            _q: [],
            on: function (e, n) {
                var A = this;
                setTimeout(function () {
                    n(A[e])
                }, 0)
            },
            addTest: function (e, n, A) {
                r.push({
                    name: e,
                    fn: n,
                    options: A
                })
            },
            addAsyncTest: function (e) {
                r.push({
                    name: null,
                    fn: e
                })
            }
        },
        Modernizr = function () {};
    Modernizr.prototype = l;
    Modernizr = new Modernizr;

    var f = [],
        u = n.documentElement,
        d = "svg" === u.nodeName.toLowerCase(),
        c = d ? u.style : u.getElementsByTagName("body")[0].style;

    Modernizr.addTest("cssanimations", function () {
        var e = "animationName",
            n = c;
        if (void 0 !== n[e]) return !0;
        var A = ["webkit", "Moz", "ms", "O"];
        for (var o = 0; o < A.length; o++)
            if (void 0 !== n[A[o] + "AnimationName"]) return !0;
        return !1
    });

    Modernizr.addTest("cssgradients", function () {
        var e = "backgroundImage",
            n = c;
        if (void 0 !== n[e] && /gradient/.test(n[e])) return !0;
        var A = ["webkit", "Moz", "ms", "O"];
        for (var o = 0; o < A.length; o++)
            if (void 0 !== n[A[o] + "BackgroundImage"] && /gradient/.test(n[A[o] + "BackgroundImage"])) return !0;
        return !1
    });

    Modernizr.addTest("csstransforms", function () {
        var e = "transform",
            n = c;
        if (void 0 !== n[e]) return !0;
        var A = ["webkit", "Moz", "ms", "O"];
        for (var o = 0; o < A.length; o++)
            if (void 0 !== n[A[o] + "Transform"]) return !0;
        return !1
    });

    Modernizr.addTest("csstransforms3d", function () {
        var e = n.createElement("p"),
            A = !!c.webkitPerspective;
        if (A) {
            var o = n.createElement("style");
            o.textContent = "@media (transform-3d),(-webkit-transform-3d){#modernizr{left:9px;position:absolute;height:3px;}}",
                u.appendChild(o),
                e.id = "modernizr",
                u.appendChild(e),
                A = 9 === e.offsetLeft && 3 === e.offsetHeight,
                u.removeChild(e),
                u.removeChild(o)
        }
        return A
    });

    Modernizr.addTest("csstransitions", function () {
        var e = "transition",
            n = c;
        if (void 0 !== n[e]) return !0;
        var A = ["webkit", "Moz", "ms", "O"];
        for (var o = 0; o < A.length; o++)
            if (void 0 !== n[A[o] + "Transition"]) return !0;
        return !1
    });

    Modernizr.addTest("flexbox", function () {
        var e = "flexWrap",
            n = c;
        if (void 0 !== n[e]) return !0;
        var A = ["webkit", "ms", "moz"];
        for (var o = 0; o < A.length; o++)
            if (void 0 !== n[A[o] + "FlexWrap"]) return !0;
        return !1
    });

    Modernizr.addTest("promises", function () {
        return "function" == typeof Promise
    });

    Modernizr.addTest("svg", function () {
        return !!n.createElementNS && !!n.createElementNS("http://www.w3.org/2000/svg", "svg").createSVGRect
    });

    s();
    a(t);
    delete l.addTest;
    delete l.addAsyncTest;

    for (var p = 0; p < Modernizr._q.length; p++)
        Modernizr._q[p]();

    e.Modernizr = Modernizr

}(window, document);
