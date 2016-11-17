copy from https://github.com/GitbookIO/plugin-katex, but support inline formula.

## Math typesetting using KaTex

Use it for your book, by adding to your book.json:

```
{
    "plugins": ["katex"]
}
```

then run `gitbook install`.

## Usage

```
Inline math: $\int_{-\infty}^\infty g(x) dx $
```

You need to put a space before `$`.

```
Block math:

$$ \int_{-\infty}^\infty g(x) dx $$
```

## input `$`

The plugin uses ` $` to indicate inline formulas. If you want to only input a `$`, you can use `\`$\``( in code mode),
