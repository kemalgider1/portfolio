import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import math


def create_portfolio_dashboard():
    # Initialize the Dash app with dark theme
    app = dash.Dash(__name__,
                    title="Portfolio Performance Dashboard",
                    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

    # Set app configuration
    app.config.suppress_callback_exceptions = True

    # Customized dark theme colors
    dark_colors = {
        'background': '#000000',
        'text': '#FFFFFF',
        'grid': '#333333',
        'kuwait_primary': '#3498db',
        'kuwait_secondary': '#2980b9',
        'jeju_primary': '#e74c3c',
        'jeju_secondary': '#c0392b',
        'kuwait_fill': 'rgba(52, 152, 219, 0.2)',
        'jeju_fill': 'rgba(231, 76, 60, 0.2)',
        'positive_text': '#2ecc71',
        'negative_text': '#e74c3c',
        'neutral_text': '#95a5a6',
        'card_bg': '#111111'
    }

    # Portfolio data structure
    portfolios = {
        'Kuwait': {
            'score': 8.26,
            'status': "HIGH PERFORMER",
            'status_color': dark_colors['kuwait_primary'],
            'fill_color': dark_colors['kuwait_fill'],
            'color': dark_colors['kuwait_primary'],
            'category_scores': {
                'Category A': 7.6,
                'Category B': 8.4,
                'Category C': 7.6,
                'Category D': 9.4
            },
            'metrics': {
                'Market Share': '48%',
                'Average Margin': 0.76,
                'Premium Mix': '70%',
                'Growth Rate': '+12.0%',
                'Total Volume': '5,300,000',
                'Green SKUs': 5,
                'Red SKUs': 2
            },
            'segments': {
                'Full Flavor': 40,
                'Light': 35,
                'Menthol': 15,
                'Ultra Light': 10
            },
            'summary': {
                'totalSKUs': 17,
                'portfolioScore': "9.42/10",
                'growthTrend': "+12.0%",
                'segmentStrength': "Full Flavor: 40%",
                'marginProfile': "0.76",
                'skuHealth': "G:5 Y:10 R:2"
            },
            'context': {
                'portfolio': 'Portfolio has 15 premium and 2 value SKUs',
                'location': 'Location is in the high performer category',
                'trend': 'Positive growth trend',
                'segment': 'Primary segment dominates portfolio',
                'margin': 'Strong premium positioning',
                'health': 'Healthy SKU portfolio'
            }
        },
        'Jeju': {
            'score': 1.97,
            'status': "REQUIRES OPTIMIZATION",
            'status_color': dark_colors['jeju_primary'],
            'fill_color': dark_colors['jeju_fill'],
            'color': dark_colors['jeju_primary'],
            'category_scores': {
                'Category A': 3.9,
                'Category B': 1.2,
                'Category C': 2.5,
                'Category D': 1.3
            },
            'metrics': {
                'Market Share': '61%',
                'Average Margin': 0.62,
                'Premium Mix': '30%',
                'Growth Rate': '-5.0%',
                'Total Volume': '2,400,000',
                'Green SKUs': 0,
                'Red SKUs': 8
            },
            'segments': {
                'Full Flavor': 70,
                'Light': 30,
                'Menthol': 0,
                'Ultra Light': 0
            },
            'summary': {
                'totalSKUs': 13,
                'portfolioScore': "1.22/10",
                'growthTrend': "-5.0%",
                'segmentStrength': "Full Flavor: 70%",
                'marginProfile': "0.62",
                'skuHealth': "G:0 Y:5 R:8"
            },
            'context': {
                'portfolio': 'Portfolio has 3 premium and 10 value SKUs',
                'location': 'Location is in the requires optimization category',
                'trend': 'Negative growth trend',
                'segment': 'Primary segment dominates portfolio',
                'margin': 'Value-oriented portfolio',
                'health': 'SKU optimization needed'
            }
        }
    }

    # Create SKU data
    sku_data = pd.DataFrame([
        # Kuwait SKUs
        {'Portfolio': 'Kuwait', 'SKU': 'MARLBORO', 'Margin': 0.85, 'Growth': 0.18, 'Volume': 700000, 'Premium': True,
         'Mix': 23.5},
        {'Portfolio': 'Kuwait', 'SKU': 'PARLIAMENT', 'Margin': 0.87, 'Growth': 0.15, 'Volume': 650000, 'Premium': True,
         'Mix': 19.4},
        {'Portfolio': 'Kuwait', 'SKU': 'HEETS', 'Margin': 0.82, 'Growth': 0.25, 'Volume': 600000, 'Premium': True,
         'Mix': 24.9},
        {'Portfolio': 'Kuwait', 'SKU': 'L&M', 'Margin': 0.72, 'Growth': 0.08, 'Volume': 550000, 'Premium': False,
         'Mix': 14.3},
        {'Portfolio': 'Kuwait', 'SKU': 'CHESTERFIELD', 'Margin': 0.90, 'Growth': 0.20, 'Volume': 500000,
         'Premium': True, 'Mix': 18.0},

        # Jeju SKUs
        {'Portfolio': 'Jeju', 'SKU': 'MARLBORO', 'Margin': 0.69, 'Growth': -0.02, 'Volume': 300000, 'Premium': True,
         'Mix': 21.5},
        {'Portfolio': 'Jeju', 'SKU': 'PARLIAMENT', 'Margin': 0.71, 'Growth': 0.01, 'Volume': 250000, 'Premium': True,
         'Mix': 17.7},
        {'Portfolio': 'Jeju', 'SKU': 'L&M', 'Margin': 0.63, 'Growth': -0.05, 'Volume': 350000, 'Premium': False,
         'Mix': 27.6},
        {'Portfolio': 'Jeju', 'SKU': 'BOND', 'Margin': 0.58, 'Growth': -0.12, 'Volume': 200000, 'Premium': False,
         'Mix': 13.3},
        {'Portfolio': 'Jeju', 'SKU': 'LARK', 'Margin': 0.60, 'Growth': -0.09, 'Volume': 300000, 'Premium': False,
         'Mix': 19.9}
    ])

    # Create radar chart function - HEXAGONAL GRAPH AS FOCUS POINT
    def create_radar_chart(portfolio_name):
        portfolio = portfolios[portfolio_name]

        categories = list(portfolio['category_scores'].keys())
        values = list(portfolio['category_scores'].values())

        # Calculate positions on a hexagon
        n_points = len(categories)
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

        # Create the radar chart
        fig = go.Figure()

        # Add circles for the radar chart gradations
        for radius in [2, 4, 6, 8, 10]:
            circle_points_x = [radius * np.cos(t) for t in np.linspace(0, 2 * np.pi, 100)]
            circle_points_y = [radius * np.sin(t) for t in np.linspace(0, 2 * np.pi, 100)]

            fig.add_trace(go.Scatter(
                x=circle_points_x,
                y=circle_points_y,
                mode='lines',
                line=dict(color='rgba(150, 150, 150, 0.3)', width=1),
                hoverinfo='skip',
                showlegend=False
            ))

        # Add radial lines
        for i in range(n_points):
            fig.add_shape(
                type='line',
                x0=0, y0=0,
                x1=10.5 * np.cos(theta[i]),
                y1=10.5 * np.sin(theta[i]),
                line=dict(color='rgba(150, 150, 150, 0.3)', width=1),
            )

        # Convert values to x,y coordinates
        x_values = [v * np.cos(t) for v, t in zip(values, theta)]
        y_values = [v * np.sin(t) for v, t in zip(values, theta)]

        # Add data trace
        fig.add_trace(go.Scatter(
            x=x_values + [x_values[0]],
            y=y_values + [y_values[0]],
            mode='lines+markers',
            line=dict(color=portfolio['color'], width=3),
            fill='toself',
            fillcolor=portfolio['fill_color'],
            marker=dict(color=portfolio['color'], size=8),
            hoverinfo='text',
            hovertext=[f"{cat}: {val}" for cat, val in zip(categories, values)],
            name=portfolio_name
        ))

        # Add category labels
        for i, (cat, val) in enumerate(zip(categories, values)):
            angle = theta[i]
            # Position labels just outside the maximum circle
            label_distance = 11

            # Add category label
            fig.add_annotation(
                x=label_distance * np.cos(angle),
                y=label_distance * np.sin(angle),
                text=f"<b>{cat}</b>",
                showarrow=False,
                font=dict(color='white', size=12),
                xanchor='center' if -0.1 < np.cos(angle) < 0.1 else 'left' if np.cos(angle) < 0 else 'right',
                yanchor='middle' if -0.1 < np.sin(angle) < 0.1 else 'bottom' if np.sin(angle) < 0 else 'top',
            )

            # Add score value inside the polygon
            score_distance = val * 0.9  # Slightly inside the point
            fig.add_annotation(
                x=score_distance * np.cos(angle),
                y=score_distance * np.sin(angle),
                text=f"<b>{val}</b>",
                showarrow=False,
                font=dict(color='white', size=12, family='Arial Black'),
                bgcolor=portfolio['color'],
                borderpad=3,
                bordercolor='white',
                borderwidth=1,
                opacity=0.8
            )

        # Update layout
        fig.update_layout(
            showlegend=False,
            plot_bgcolor=dark_colors['background'],
            paper_bgcolor=dark_colors['background'],
            margin=dict(l=10, r=10, t=40, b=10),
            height=400,
            xaxis=dict(
                showgrid=False, zeroline=False, visible=False,
                range=[-12, 12]
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, visible=False,
                range=[-12, 12], scaleanchor="x", scaleratio=1
            ),
            title=dict(
                text="Category Scores",
                font=dict(color=dark_colors['text'], size=16),
                x=0.5
            ),
            annotations=[
                dict(
                    x=0, y=0,
                    text="10",
                    font=dict(color="white", size=10),
                    showarrow=False,
                    xshift=40,
                    yshift=0
                ),
                dict(
                    x=0, y=0,
                    text="8",
                    font=dict(color="white", size=10),
                    showarrow=False,
                    xshift=32,
                    yshift=0
                ),
                dict(
                    x=0, y=0,
                    text="6",
                    font=dict(color="white", size=10),
                    showarrow=False,
                    xshift=24,
                    yshift=0
                ),
                dict(
                    x=0, y=0,
                    text="4",
                    font=dict(color="white", size=10),
                    showarrow=False,
                    xshift=16,
                    yshift=0
                ),
                dict(
                    x=0, y=0,
                    text="2",
                    font=dict(color="white", size=10),
                    showarrow=False,
                    xshift=8,
                    yshift=0
                )
            ]
        )

        return fig

    # Create segment distribution chart
    def create_segment_distribution(portfolio_name):
        portfolio = portfolios[portfolio_name]

        # Convert segments to DataFrame
        segments = pd.DataFrame([(k, v) for k, v in portfolio['segments'].items()],
                                columns=['Segment', 'Percentage'])

        # Ensure the order is correct
        segment_order = ['Full Flavor', 'Light', 'Menthol', 'Ultra Light']
        segments['Segment'] = pd.Categorical(segments['Segment'], categories=segment_order, ordered=True)
        segments = segments.sort_values('Segment')

        fig = px.bar(
            segments,
            x='Percentage',
            y='Segment',
            orientation='h',
            text='Percentage',
            color_discrete_sequence=[portfolio['color']]
        )

        fig.update_traces(
            texttemplate='%{text}%',
            textposition='outside',
            cliponaxis=False,
            marker_line_width=0
        )

        fig.update_layout(
            title=dict(
                text="Segment Distribution",
                font=dict(color=dark_colors['text'], size=16),
                x=0.5
            ),
            plot_bgcolor=dark_colors['background'],
            paper_bgcolor=dark_colors['background'],
            font=dict(color=dark_colors['text']),
            margin=dict(l=10, r=10, t=40, b=10),
            height=400,
            xaxis=dict(
                title="Share of Portfolio",
                range=[0, 100],
                showgrid=True,
                gridcolor=dark_colors['grid'],
                ticksuffix="%",
                zeroline=False
            ),
            yaxis=dict(
                title=None,
                autorange="reversed",
                showgrid=False,
                zeroline=False
            )
        )

        return fig

    # Create SKU performance matrix
    def create_sku_matrix(portfolio_name):
        portfolio_data = sku_data[sku_data['Portfolio'] == portfolio_name]

        # Set colors for premium and non-premium
        if portfolio_name == 'Kuwait':
            premium_color = dark_colors['kuwait_primary']
            nonpremium_color = dark_colors['kuwait_secondary']
        else:
            premium_color = dark_colors['jeju_primary']
            nonpremium_color = dark_colors['jeju_secondary']

        # Create custom hover text
        hover_text = []
        for index, row in portfolio_data.iterrows():
            hover_text.append(
                f"<b>{row['SKU']}</b><br>" +
                f"Margin: {row['Margin']:.2f}<br>" +
                f"Growth: {row['Growth']:.1%}<br>" +
                f"Volume: {row['Volume']:,}"
            )

        # Create scatter plot
        fig = go.Figure()

        # Add premium SKUs
        premium_data = portfolio_data[portfolio_data['Premium']]
        fig.add_trace(go.Scatter(
            x=premium_data['Margin'],
            y=premium_data['Growth'],
            mode='markers',
            marker=dict(
                color=premium_color,
                size=premium_data['Volume'].apply(lambda x: max(10, min(np.sqrt(x / 10000), 40))),
                line=dict(width=1, color='white')
            ),
            text=premium_data['SKU'],
            hoverinfo='text',
            hovertext=[hover_text[i] for i in premium_data.index],
            name='Premium'
        ))

        # Add non-premium SKUs
        nonpremium_data = portfolio_data[~portfolio_data['Premium']]
        fig.add_trace(go.Scatter(
            x=nonpremium_data['Margin'],
            y=nonpremium_data['Growth'],
            mode='markers',
            marker=dict(
                color=nonpremium_color,
                size=nonpremium_data['Volume'].apply(lambda x: max(10, min(np.sqrt(x / 10000), 40))),
                line=dict(width=1, color='white')
            ),
            text=nonpremium_data['SKU'],
            hoverinfo='text',
            hovertext=[hover_text[i] for i in nonpremium_data.index],
            name='Value'
        ))

        # Add SKU labels
        for i, row in portfolio_data.iterrows():
            fig.add_annotation(
                x=row['Margin'],
                y=row['Growth'],
                text=row['SKU'],
                showarrow=False,
                font=dict(size=10, color='white'),
                bgcolor='rgba(0,0,0,0.5)',
                borderpad=2
            )

        # Add quadrant lines
        fig.add_shape(
            type="line", x0=0.75, y0=-0.15, x1=0.75, y1=0.3,
            line=dict(color="grey", width=1, dash="dash")
        )
        fig.add_shape(
            type="line", x0=0.5, y0=0, x1=0.95, y1=0,
            line=dict(color="grey", width=1, dash="dash")
        )

        # Add quadrant labels
        fig.add_annotation(x=0.85, y=0.15, text="Premium Growers",
                           showarrow=False, font=dict(color='white', size=10),
                           bgcolor="rgba(0,0,0,0.5)")
        fig.add_annotation(x=0.65, y=0.15, text="Value Growers",
                           showarrow=False, font=dict(color='white', size=10),
                           bgcolor="rgba(0,0,0,0.5)")
        fig.add_annotation(x=0.65, y=-0.07, text="Underperformers",
                           showarrow=False, font=dict(color='white', size=10),
                           bgcolor="rgba(0,0,0,0.5)")
        fig.add_annotation(x=0.85, y=-0.07, text="Premium Decliners",
                           showarrow=False, font=dict(color='white', size=10),
                           bgcolor="rgba(0,0,0,0.5)")

        # Update layout
        fig.update_layout(
            title=dict(
                text="SKU Performance Matrix (Equal Volume Visualization)",
                font=dict(color=dark_colors['text'], size=16),
                x=0.5
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                font=dict(color=dark_colors['text']),
                bgcolor='rgba(0,0,0,0)'
            ),
            plot_bgcolor=dark_colors['background'],
            paper_bgcolor=dark_colors['background'],
            margin=dict(l=10, r=10, t=40, b=10),
            height=400,
            xaxis=dict(
                title="Margin",
                showgrid=True,
                gridcolor=dark_colors['grid'],
                zeroline=False,
                range=[
                    portfolio_data['Margin'].min() - 0.05,
                    portfolio_data['Margin'].max() + 0.05
                ]
            ),
            yaxis=dict(
                title="Year-over-Year Growth",
                showgrid=True,
                gridcolor=dark_colors['grid'],
                zeroline=True,
                zerolinecolor=dark_colors['grid'],
                range=[
                    min(portfolio_data['Growth'].min() - 0.05, -0.15),
                    max(portfolio_data['Growth'].max() + 0.05, 0.3)
                ]
            )
        )

        return fig

    # Create Brand Mix chart
    def create_brand_mix(portfolio_name):
        portfolio_data = sku_data[sku_data['Portfolio'] == portfolio_name]

        # Determine colors
        if portfolio_name == 'Kuwait':
            colors = px.colors.sequential.Blues
        else:
            colors = px.colors.sequential.Reds

        # Create pie chart
        fig = px.pie(
            portfolio_data,
            values='Mix',
            names='SKU',
            color_discrete_sequence=colors,
            hole=0.4
        )

        fig.update_traces(
            textposition='inside',
            textinfo='label+percent',
            marker=dict(line=dict(color=dark_colors['background'], width=1))
        )

        fig.update_layout(
            title=dict(
                text="Brand Mix by Volume",
                font=dict(color=dark_colors['text'], size=16),
                x=0.5
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.2,
                xanchor='center',
                x=0.5,
                font=dict(color=dark_colors['text']),
                bgcolor='rgba(0,0,0,0)'
            ),
            plot_bgcolor=dark_colors['background'],
            paper_bgcolor=dark_colors['background'],
            margin=dict(l=10, r=10, t=40, b=10),
            height=400
        )

        return fig

    # Create metrics table
    def create_metrics_table(portfolio_name):
        portfolio = portfolios[portfolio_name]
        metrics = portfolio['metrics']

        # Create metrics table with styling based on value type
        table_data = []
        colors = []

        for metric, value in metrics.items():
            table_data.append([metric, value])

            # Determine cell color based on metric and value
            if metric == 'Growth Rate':
                if isinstance(value, str) and '+' in value:
                    colors.append(dark_colors['positive_text'])
                else:
                    colors.append(dark_colors['negative_text'])
            elif metric == 'Average Margin':
                if portfolio_name == 'Kuwait':
                    colors.append(dark_colors['positive_text'])
                else:
                    colors.append(dark_colors['neutral_text'])
            elif metric == 'Premium Mix':
                if portfolio_name == 'Kuwait':
                    colors.append(dark_colors['positive_text'])
                else:
                    colors.append(dark_colors['negative_text'])
            elif metric == 'Green SKUs':
                colors.append(dark_colors['positive_text'])
            elif metric == 'Red SKUs':
                if portfolio_name == 'Kuwait':
                    colors.append(dark_colors['positive_text'])
                else:
                    colors.append(dark_colors['negative_text'])
            else:
                colors.append(dark_colors['text'])

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Key Performance Metrics</b>', '<b>Value</b>'],
                fill_color='#333333',
                font=dict(color=dark_colors['text'], size=14),
                align='left',
                height=30
            ),
            cells=dict(
                values=list(zip(*table_data)),
                fill_color=dark_colors['background'],
                font=dict(color=[dark_colors['text'], colors], size=14),
                align='left',
                height=25
            )
        )])

        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=350,
            plot_bgcolor=dark_colors['background'],
            paper_bgcolor=dark_colors['background']
        )

        return fig

    # Create score card
    def create_score_card(portfolio_name):
        portfolio = portfolios[portfolio_name]
        score = portfolio['score']
        status = portfolio['status']

        fig = go.Figure()

        # Add title
        fig.add_trace(go.Indicator(
            mode="number",
            value=score,
            number=dict(
                font=dict(color=portfolio['status_color'], size=60),
                suffix="/10"
            ),
            title=dict(
                text=f"<b>Score: {score}/10 - {status}</b>",
                font=dict(color=dark_colors['text'], size=16)
            ),
            domain=dict(x=[0, 1], y=[0, 1])
        ))

        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=120,
            plot_bgcolor=dark_colors['background'],
            paper_bgcolor=dark_colors['background']
        )

        return fig

    # Create portfolio summary
    def create_summary_table(portfolio_name):
        portfolio = portfolios[portfolio_name]
        summary = portfolio['summary']
        context = portfolio['context']

        table_data = []
        for key in summary:
            if key in context:
                table_data.append([key.replace('total', 'Total').replace('portfolio', 'Portfolio').replace('growth',
                                                                                                           'Growth').replace(
                    'segment', 'Segment').replace('margin', 'Margin').replace('sku', 'SKU'),
                                   summary[key], context[key]])

        # Add colors based on portfolio
        if portfolio_name == 'Kuwait':
            value_color = dark_colors['kuwait_primary']
        else:
            value_color = dark_colors['jeju_primary']

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Metric</b>', '<b>Value</b>', '<b>Context</b>'],
                fill_color='#333333',
                font=dict(color=dark_colors['text'], size=14),
                align='center',
                height=30
            ),
            cells=dict(
                values=list(zip(*table_data)),
                fill_color=['#222222', '#1a1a1a'],
                font_color=[dark_colors['text'], value_color, dark_colors['text']],
                align='center',
                height=30
            )
        )])

        fig.update_layout(
            title=dict(
                text="Portfolio Performance Summary",
                font=dict(color=dark_colors['text'], size=16),
                x=0.5
            ),
            margin=dict(l=10, r=10, t=40, b=10),
            height=300,
            plot_bgcolor=dark_colors['background'],
            paper_bgcolor=dark_colors['background']
        )

        return fig

    # App layout
    app.layout = html.Div([
        # Title and Portfolio selector
        html.Div([
            html.H1("PORTFOLIO PERFORMANCE",
                    style={'textAlign': 'center', 'color': dark_colors['text'], 'paddingTop': '20px'}),
            html.Div([
                dcc.RadioItems(
                    id='portfolio-selector',
                    options=[
                        {'label': 'Kuwait', 'value': 'Kuwait'},
                        {'label': 'Jeju', 'value': 'Jeju'}
                    ],
                    value='Kuwait',
                    labelStyle={'display': 'inline-block', 'marginRight': '20px', 'color': dark_colors['text']}
                )
            ], style={'textAlign': 'center', 'padding': '10px'})
        ], style={'backgroundColor': dark_colors['background']}),

        # Main Content
        html.Div([
            # Portfolio Score Card
            html.Div([
                html.Div(id='score-card')
            ], className='row'),

            # First row - Category Radar Chart and Segment Distribution
            html.Div([
                html.Div([
                    dcc.Graph(id='category-radar', config={'displayModeBar': False})
                ], className='col-md-6'),
                html.Div([
                    dcc.Graph(id='segment-chart', config={'displayModeBar': False})
                ], className='col-md-6')
            ], className='row'),

            # Second row - Key Performance Metrics and Brand Mix
            html.Div([
                html.Div([
                    dcc.Graph(id='metrics-table', config={'displayModeBar': False})
                ], className='col-md-6'),
                html.Div([
                    dcc.Graph(id='brand-mix', config={'displayModeBar': False})
                ], className='col-md-6')
            ], className='row'),

            # Third row - SKU Matrix
            html.Div([
                html.Div([
                    dcc.Graph(id='sku-matrix', config={'displayModeBar': False})
                ], className='col-md-12')
            ], className='row'),

            # Fourth row - Summary Table
            html.Div([
                html.Div([
                    dcc.Graph(id='summary-table', config={'displayModeBar': False})
                ], className='col-md-12')
            ], className='row'),

            # Footer
            html.Div([
                html.P(f"Generated on 2025-02-28 | Portfolio Analysis",
                       style={'textAlign': 'center', 'color': dark_colors['neutral_text'], 'padding': '20px'})
            ])
        ], style={'backgroundColor': dark_colors['background'], 'padding': '0 20px'}),

    ], style={'backgroundColor': dark_colors['background'], 'fontFamily': 'Arial, sans-serif'})

    # Define callbacks to update charts based on selected portfolio
    @app.callback(
        [Output('score-card', 'children'),
         Output('category-radar', 'figure'),
         Output('segment-chart', 'figure'),
         Output('metrics-table', 'figure'),
         Output('brand-mix', 'figure'),
         Output('sku-matrix', 'figure'),
         Output('summary-table', 'figure')],
        [Input('portfolio-selector', 'value')]
    )
    def update_dashboard(selected_portfolio):
        # Create portfolio title with portfolio-specific styling
        title_style = {
            'textAlign': 'center',
            'color': 'white',
            'padding': '10px',
            'fontSize': '24px',
            'fontWeight': 'bold',
            'backgroundColor': portfolios[selected_portfolio]['status_color']
        }

        score_card = html.Div([
            html.H2(f"PORTFOLIO PERFORMANCE: {selected_portfolio}", style=title_style),
            dcc.Graph(figure=create_score_card(selected_portfolio), config={'displayModeBar': False})
        ])

        # Generate all charts
        category_radar = create_radar_chart(selected_portfolio)
        segment_chart = create_segment_distribution(selected_portfolio)
        metrics_table = create_metrics_table(selected_portfolio)
        brand_mix = create_brand_mix(selected_portfolio)
        sku_matrix = create_sku_matrix(selected_portfolio)
        summary_table = create_summary_table(selected_portfolio)

        return score_card, category_radar, segment_chart, metrics_table, brand_mix, sku_matrix, summary_table

    # Add custom CSS for dark theme
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    background-color: #000000;
                    color: #FFFFFF;
                    margin: 0;
                    padding: 0;
                }
                .dash-graph {
                    border-radius: 5px;
                    background-color: #111111;
                    margin-bottom: 15px;
                }
                .dash-table-container {
                    background-color: #111111;
                    border-radius: 5px;
                }
                .row {
                    margin-bottom: 20px;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''

    return app


# Create and run the app
if __name__ == '__main__':
    dashboard_app = create_portfolio_dashboard()
    dashboard_app.run_server(debug=True)