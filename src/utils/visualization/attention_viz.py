from codecs import open

import matplotlib
matplotlib.use('Agg')  # or 'PS', 'PDF', 'SVG'

import matplotlib.pyplot as plt
import numpy as np


def createHTML(texts, att_heads, weights, export_html):
    """
    ### Credits to Lin Zhouhan(@hantek) for this visualization code

    Creates a html file with text heat.
    att_heads: attention heads the examples are assigned to
    weights: attention weights for visualizing
    texts: text on which attention weights are to be visualized
    """

    html_file = open(export_html, 'w', encoding='utf-8')

    part1 = """
    <html lang="en">
    <head>
    <meta http-equiv="content-type" content="text/html; charset=utf-8">
    <style>
    body {
    font-family: Sans-Serif;
    }
    </style>
    </head>
    <body>
    <h3>
    Visualization of Self-Attention Weights
    </h3>
    </body>
    <script>
    """

    part2 = """
    var color = "255,0,0";
    var ngram_length = 3;
    var half_ngram = 1;
    for (var k=0; k < any_text.length; k++) {
    var tokens = any_text[k].split(" ");
    var intensity = new Array(tokens.length);
    var max_intensity = Number.MIN_SAFE_INTEGER;
    var min_intensity = Number.MAX_SAFE_INTEGER;
    for (var i = 0; i < intensity.length; i++) {
    intensity[i] = 0.0;
    for (var j = -half_ngram; j < ngram_length-half_ngram; j++) {
    if (i+j < intensity.length && i+j > -1) {
    intensity[i] += trigram_weights[k][i + j];
    }
    }
    if (i == 0 || i == intensity.length-1) {
    intensity[i] /= 2.0;
    } else {
    intensity[i] /= 3.0;
    }
    if (intensity[i] > max_intensity) {
    max_intensity = intensity[i];
    }
    if (intensity[i] < min_intensity) {
    min_intensity = intensity[i];
    }
    }
    var denominator = max_intensity - min_intensity;
    for (var i = 0; i < intensity.length; i++) {
    intensity[i] = (intensity[i] - min_intensity) / denominator;
    }
    if (k%2 == 0) {
    var heat_text = "<p><br><b>" + example_range[k] + ". (h" + att_heads[k] + ")</b><br>";
    } else {
    var heat_text = "<b>" + example_range[k] + ". (h" + att_heads[k] + ")</b><br>";
    }
    var space = "";
    for (var i = 0; i < tokens.length; i++) {
    heat_text += "<span style='background-color:rgba(" + color + "," + intensity[i] + ")'>" + space + tokens[i] + "</span>";
    if (space == "") {
    space = " ";
    }
    }
    //heat_text += "<p>";
    document.body.innerHTML += heat_text;
    }
    </script>
    </html>"""

    set_quote = lambda x: '\"%s\"' % x
    texts_str = 'var any_text = [%s];\n' % (','.join(map(set_quote, texts)))
    att_heads_str = 'var att_heads = [%s];\n' % (','.join(map(str, att_heads)))
    weights_str = 'var trigram_weights = [%s];\n' % (','.join(map(str, weights)))
    range_str = 'var example_range = [%s];\n' % (','.join(map(str, range(1, len(att_heads)+1))))

    html_file.write(part1)
    html_file.write(texts_str)
    html_file.write(att_heads_str)
    html_file.write(weights_str)
    html_file.write(range_str)
    html_file.write(part2)
    html_file.close()

    return


def plot_matrix_heatmap(a, title='', export_pdf=False, show=False):
    """ Plots heatmap for matrix a. """

    a = np.round(a, decimals=2)

    ax = plt.gca()
    # plot the heatmap
    im = ax.imshow(a, cmap='RdYlBu')
    # create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('color gradient', rotation=-90, va='bottom')

    # Loop over data and create value annotations
    n, m = a.shape
    for i in range(n):
        for j in range(m):
            im.axes.text(j, i, a[i, j], ha='center', va='center', color='black')

    plt.title(title)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if export_pdf:
        plt.savefig(export_pdf, bbox_inches='tight', pad_inches=0.1)
        plt.clf()

    if show:
        plt.show()

    return
