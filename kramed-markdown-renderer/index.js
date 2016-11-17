function identity(x) {
    return x;
}

function reverse(str) {
    return str
    .split('')
    .reverse()
    .join('');
}

function wrap(str, wrapper) {
    return wrapper + str + reverse(wrapper);
}

function repeat(str, n) {
    var s = '';
    for(var i = n; n--; n <= 0) {
        s += str;
    }
    return s;
}

function block(str) {
    return wrap(str.trim(), '\n');
}

function lines(str) {
    var s = str.trim();
    return (s ? s.split('\n') : []);
}

function indent(str, prefix) {
    // Return empty strings
    prefix = prefix || '    ';
    return lines(str)
    .map(function(line) {
        return prefix + line;
    })
    .join('\n');
}

function MarkdownRenderer(options) {
    if(!(this instanceof MarkdownRenderer)) {
        return new MarkdownRenderer(options);
    }
    this.options = options || {};
}

MarkdownRenderer.prototype.code = function(code, lang, escaped) {
    return block(
        wrap(lang + block(code), '```')
    );
};

MarkdownRenderer.prototype.blockquote = function(quote) {
    return block(indent(quote, '> '));
};

MarkdownRenderer.prototype.html = function(html) {
    return block(html);
};

MarkdownRenderer.prototype.heading = function(text, level, raw) {
    return block(repeat('#', level) + ' ' + raw);
};

MarkdownRenderer.prototype.hr = function() {
    return block('---');
};

MarkdownRenderer.prototype._listOrder = function(body, ordered) {
    if(!ordered) {
        return body;
    }
    return lines(body)
    .map(function(line, idx) {
        return line.replace(/^\* /, (idx+1)+'. ');
    })
    .join('\n');
};

MarkdownRenderer.prototype.list = function(body, ordered) {
    return block(this._listOrder(body, ordered));
};

MarkdownRenderer.prototype.listitem = function(text) {
    var rows = lines(text);
    var head = rows[0];
    var rest = indent(rows.slice(1).join('\n'));
    return '\n' + '* ' + head + (rest ? '\n' + rest : '');
};

MarkdownRenderer.prototype.paragraph = function(text) {
    return block(text);
};

MarkdownRenderer.prototype.table = function(header, body) {
    body = body.trim();
    header = header.trim();

    // Patch cell rows
    var patch = function(str) {
        return indent(str, '| ');
    };

    // Build seperator between header and body
    var headerLine = patch(header)
    .split('|')
    .map(function(headerRow) {
        return headerRow.length > 2 ? wrap(repeat('-', headerRow.length - 2), ' ') : ''
    })
    .join('|');

    return block(patch(header)) + headerLine + block(patch(body));
};

MarkdownRenderer.prototype.tablerow = function(content) {
    return content + '\n';
};

MarkdownRenderer.prototype.tablecell = function(content, flags) {
    return content + ' | ';
};

// span level renderer
MarkdownRenderer.prototype.strong = function(text) {
    return wrap(text, '**');
};

MarkdownRenderer.prototype.em = function(text) {
    return wrap(text, '*');
};

MarkdownRenderer.prototype.codespan = function(text) {
    return wrap(text, '`');
};

MarkdownRenderer.prototype.br = function() {
    return '\n\n';
};

MarkdownRenderer.prototype.del = function(text) {
    return text;
};

MarkdownRenderer.prototype.link = function(href, title, text) {
    // Detect and handle inline links
    if(text == href) {
        return href;
    }
    return '['+text+']('+href+')';
};

MarkdownRenderer.prototype.image = function(href, title, text) {
    return '!'+MarkdownRenderer.prototype.link(href, title, text);
};


MarkdownRenderer.prototype.footnote = function(refname, text) {
    return '[^'+refname+']: ' + text;
};

MarkdownRenderer.prototype.math = function(formula, type, inline) {
    var wrapper = inline ? identity : block;
    return wrapper('$$'+wrapper(formula)+'$$');
};

MarkdownRenderer.prototype.reffn = function(refname) {
    return '[^'+refname+']';
};

// Exports
module.exports = MarkdownRenderer;
