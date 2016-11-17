var katex = require("katex");

module.exports = {
    book: {
        assets: "./static",
        js: [],
        css: [
            "katex.min.css"
        ]
    },
    ebook: {
        assets: "./static",
        css: [
            "katex.min.css"
        ]
    },
    blocks: {
        display_math: {
            shortcuts: {
                parsers: ["markdown"],
                start: "$$",
                end: "$$"
            },
            process: function(blk) {
                var tex = blk.body;
                var output = katex.renderToString(tex, {
                    displayMode: true
                });

                return output;
            }
        },
        inline_math: {
            shortcuts: {
                parsers: ["markdown"],
                start: " $",
                end: " $"
            },
            process: function(blk) {
                var tex = blk.body;
                console.log(tex);
                tex = tex.replace(/\\\(/g, '(').replace(/\\\)/g, ")");
                console.log(tex);
                var output = katex.renderToString(tex, {
                    displayMode: false
                });

                console.log(output)

                return output;
            }
        },
    },
    hooks: {
        "page": function(section) {
            section.content += '<script src="//cdn.bootcss.com/mathjax/2.7.0-beta.0/MathJax.js"></script>';

            return section;
        }
    }
};
