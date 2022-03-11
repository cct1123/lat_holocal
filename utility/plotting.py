import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
# ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none']
pio.templates.default = "plotly_dark"

def scalar_field_plot(x, y, field, 
                      normalize = False,
                      title = 'Beam',
                      plot1title = "Phase [rad]",
                      plot2title = "Intensity [dB]",
                      xlabel = "x[a.u.]",
                      ylabel = "y[a.u.]",
                      cmap1 = 'Twilight',
                      cmap2 = 'Magma', 
                     ):
    # title = 'Beam on sky'
    # plot1title = "Phase [rad]"
    # plot2title = "Intensity [dB]"
    # xlabel = "el [mrad]"
    # ylabel = "az [mrad]"
    phase = np.angle(field)
    phase = np.mod(phase, 2*np.pi)
    if normalize:
        amp = np.abs(field)
        amp = 20*np.log10(amp/np.max(amp))
    else:
        amp = 20*np.log10(np.abs(field))

    plot_1 = go.Heatmap(
                        x=x, 
                        y=y,
                        z = phase,
                        colorscale=cmap1,
                        showscale=True, colorbar=dict(len=0.8, x=0.44),
                        showlegend = False,
                        # hoverinfo='name', 
                        name="phase"
                        )
    plot_2 = go.Heatmap( 
                        x=x, 
                        y=y,
                        z = amp,
                        # z= np.abs(field_fp*field_fp),
                        colorscale=cmap2,
                        showscale=True, 
                        colorbar=dict(len=0.8, x=1), 
                        showlegend = False,
                        # hoverinfo='name', 
                        name="power"
                        )
    layout = go.Layout(title=title, autosize=True,
                    width=1200, height=500, 
                    margin=dict(l=50, r=50, b=65, t=90),
                    xaxis1 = dict(title=xlabel, showgrid=False, zeroline=False),
                    yaxis1 = dict(title=ylabel, scaleanchor = 'x', showgrid=False, zeroline=False),
                    xaxis2 = dict(title=xlabel, scaleanchor = "x", showgrid=False, zeroline=False),
                    yaxis2 = dict(scaleanchor = "y", showgrid=False, zeroline=False),
                    # xaxis1 = dict(title=xlabel),
                    # yaxis1 = dict(title=ylabel, scaleanchor = 'x'),
                    # xaxis2 = dict(title=xlabel, scaleanchor = "x"),
                    # yaxis2 = dict(scaleanchor = "y"),
                    # paper_bgcolor="#305480"
                    )

    fig = make_subplots(rows=1, cols=2, shared_yaxes=False, shared_xaxes=True, subplot_titles=[plot1title, plot2title])
    fig.add_trace(plot_1, row=1, col=1)
    fig.add_trace(plot_2, row=1, col=2)
    fig.update_layout(layout)
    fig.show()

if __name__ == "__main__":
    x = np.linspace(-7*np.pi, 7*np.pi, 200)
    y = np.linspace(-2*np.pi, 2*np.pi, 200)
    xmesh, ymesh = np.meshgrid(x, y)
    field = np.cos(ymesh*ymesh+xmesh*xmesh)*np.exp(1j*(xmesh+y))
    title = 'Dummy Beam'
    plot1title = "Phase [rad]"
    plot2title = "Intensity [dB]"
    xlabel = "x [rad]"
    ylabel = "y [rad]"
    scalar_field_plot(x, y, field, 
                      title = title,
                      plot1title = plot1title,
                      plot2title = plot2title,
                      xlabel = xlabel,
                      ylabel = ylabel)