var fs = require('fs');
var path = require('path');
var assert = require('assert');

var kramed = require('kramed');

var renderer = require('../');

var CONTENT = fs.readFileSync(path.join(__dirname, './fixtures/PAGE.md'), 'utf8');
var RENDERED = render(CONTENT);

function render(content) {
    var lexed = kramed.lexer(content);

    // Options to parser
    var options = Object.create(kramed.defaults);
    options.renderer = renderer();

    return kramed.parser(lexed, options);
}

describe('Markdown renderer', function() {
    it('should strip all html tags', function() {
        assert.equal(RENDERED.indexOf('</'), -1);
    });
    it('should produce the same html output as the original', function() {
        assert.equal(
            kramed(RENDERED),
            kramed(CONTENT)
        );
    });
});
