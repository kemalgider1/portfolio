import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output


def create_portfolio_dashboard():
    # Initialize the app
    app = dash.Dash(__name__, title="Portfolio Performance Dashboard")

    # Define portfolio data
    portfolios = {
        'kuwait': {
            'name': 'Kuwait',
            'score': 8.26,
            'status': "HIGH PERFORMER",
            'color_scheme': {
                'primary': "#3498db",
                'secondary': "#2980b9",
                'accent': "#1abc9c",
                'background': "#f8fafc"
            },
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
            'skus': [
                {'sku': 'MARLBORO', 'margin': 0.85, 'growth': 0.18, 'volume': 700000, 'premium': True},
                {'sku': 'PARLIAMENT', 'margin': 0.87, 'growth': 0.15, 'volume': 650000, 'premium': True},
                {'sku': 'HEETS', 'margin': 0.82, 'growth': 0.25, 'volume': 600000, 'premium': True},
                {'sku': 'L&M', 'margin': 0.72, 'growth': 0.08, 'volume': 550000, 'premium': False},
                {'sku': 'CHESTERFIELD', 'margin': 0.90, 'growth': 0.20, 'volume': 500000, 'premium': True}
            ],
            'brand_mix': [
                {'name': 'MARLBORO', 'value': 23.5},
                {'name': 'PARLIAMENT', 'value': 19.4},
                {'name': 'HEETS', 'value': 24.9},
                {'name': 'L&M', 'value': 14.3},
                {'name': 'CHESTERFIELD', 'value': 18.0}
            ],
            'summary': {
                'totalSKUs': 17,
                'portfolioScore': "8.26/10",
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
        'jeju': {
            'name': 'Jeju',
            'score': 1.97,
            'status': "REQUIRES OPTIMIZATION",
            'color_scheme': {
                'primary': "#e74c3c",
                'secondary': "#c0392b",
                'accent': "#e67e22",
                'background': "#f8fafc"
            },
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
            'skus': [
                {'sku': 'MARLBORO', 'margin': 0.69, 'growth': -0.02, 'volume': 300000, 'premium': True},
                {'sku': 'PARLIAMENT', 'margin': 0.71, 'growth': 0.01, 'volume': 250000, 'premium': True},
                {'sku': 'L&M', 'margin': 0.63, 'growth': -0.05, 'volume': 350000, 'premium': False},
                {'sku': 'BOND', 'margin': 0.58, 'growth': -0.12, 'volume': 200000, 'premium': False},
                {'sku': 'LARK', 'margin': 0.60, 'growth': -0.09, 'volume': 300000, 'premium': False}
            ],
            'brand_mix': [
                {'name': 'MARLBORO', 'value': 21.5},
                {'name': 'PARLIAMENT', 'value': 17.7},
                {'name': 'L&M', 'value': 27.6},
                {'name': 'BOND', 'value': 13.3},
                {'name': 'LARK', 'value': 19.9}
            ],
            'summary': {
                'totalSKUs': 13,
                'portfolioScore': "1.97/10",
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

    # Create header with score indicator
    def create_header(portfolio_key):
        portfolio = portfolios[portfolio_key]

        fig = go.Figure()

        # Add title and score
        fig.add_trace(go.Indicator(
            mode="number",
            value=portfolio['score'],
            number={
                "font": {"size": 80, "color": portfolio['color_scheme']['primary']},
                "suffix": "/10"
            },
            title={
                "text": f"<b>PORTFOLIO PERFORMANCE: {portfolio['name'].upper()}</b><br><span style='font-size:1.2em;color:{portfolio['color_scheme']['primary']}'>{portfolio['status']}</span>",
                "font": {"size": 24}
            },
            domain={"row": 0, "column": 0}
        ))

        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=30, b=30),
            paper_bgcolor=portfolio['color_scheme']['background'],
            plot_bgcolor=portfolio['color_scheme']['background']
        )

        return fig

    # Create radar chart for category scores
    def create_radar_chart(portfolio_key):
        portfolio = portfolios[portfolio_key]
        categories = list(portfolio['category_scores'].keys())
        values = list(portfolio['category_scores'].values())

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=portfolio['name'],
            line=dict(color=portfolio['color_scheme']['primary'], width=2),
            fillcolor=f"rgba({int(portfolio['color_scheme']['primary'][1:3], 16)}, {int(portfolio['color_scheme']['primary'][3:5], 16)}, {int(portfolio['color_scheme']['primary'][5:7], 16)}, 0.2)"
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    tickfont=dict(size=10),
                    tickvals=[2, 4, 6, 8, 10]
                ),
                angularaxis=dict(
                    tickfont=dict(size=12, color="black")
                )
            ),
            showlegend=False,
            title={
                'text': "Category Scores",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            height=400,
            margin=dict(l=80, r=80, t=50, b=50),
            paper_bgcolor=portfolio['color_scheme']['background'],
            plot_bgcolor=portfolio['color_scheme']['background']
        )

        # Add score annotations
        for i, score in enumerate(values):
            angle = (i * 2 * np.pi / len(categories))
            r_offset = 0.5
            x = (score + r_offset) * np.cos(angle)
            y = (score + r_offset) * np.sin(angle)

            fig.add_annotation(
                x=x, y=y,
                text=f"{score}",
                font=dict(size=12, color=portfolio['color_scheme']['primary'], family="Arial Black"),
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                bgcolor="white",
                bordercolor=portfolio['color_scheme']['primary'],
                borderwidth=1,
                borderpad=3,
                opacity=0.8
            )

        return fig

    # Create segment distribution chart
    def create_segment_chart(portfolio_key):
        portfolio = portfolios[portfolio_key]
        segments = pd.DataFrame(
            [(k, v) for k, v in portfolio['segments'].items()],
            columns=['Segment', 'Percentage']
        )

        fig = px.bar(
            segments,
            y='Segment',
            x='Percentage',
            orientation='h',
            color_discrete_sequence=[portfolio['color_scheme']['primary']],
            text='Percentage'
        )

        fig.update_traces(
            texttemplate='%{text}%',
            textposition='outside',
            marker_line_width=0
        )

        fig.update_layout(
            title={
                'text': "Segment Distribution",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Share of Portfolio (%)",
            xaxis=dict(range=[0, 100]),
            yaxis=dict(categoryorder='array', categoryarray=segments['Segment'].tolist()),
            height=400,
            margin=dict(l=20, r=20, t=50, b=50),
            paper_bgcolor=portfolio['color_scheme']['background'],
            plot_bgcolor=portfolio['color_scheme']['background']
        )

        return fig

    # Create SKU performance matrix
    def create_sku_matrix(portfolio_key):
        portfolio = portfolios[portfolio_key]

        # Convert SKU data to DataFrame
        sku_df = pd.DataFrame(portfolio['skus'])

        # Create the scatter plot
        fig = px.scatter(
            sku_df,
            x='margin',
            y='growth',
            size='volume',
            size_max=40,
            color='premium',
            color_discrete_map={True: portfolio['color_scheme']['primary'],
                                False: portfolio['color_scheme']['secondary']},
            hover_name='sku',
            custom_data=['sku', 'volume']
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
        fig.add_annotation(x=0.85, y=0.15, text="Premium Growers", showarrow=False, bgcolor="rgba(255,255,255,0.7)")
        fig.add_annotation(x=0.65, y=0.15, text="Value Growers", showarrow=False, bgcolor="rgba(255,255,255,0.7)")
        fig.add_annotation(x=0.65, y=-0.07, text="Underperformers", showarrow=False, bgcolor="rgba(255,255,255,0.7)")
        fig.add_annotation(x=0.85, y=-0.07, text="Premium Decliners", showarrow=False, bgcolor="rgba(255,255,255,0.7)")

        # Add optimization suggestions for Jeju
        if portfolio_key == 'jeju':
            fig.add_annotation(
                x=0.77, y=0.1,
                text="Opportunity Area: Premium Growth",
                font=dict(color="#e74c3c"),
                bgcolor="rgba(255,255,255,0.7)",
                showarrow=True,
                arrowhead=1
            )

        # Update layout
        fig.update_layout(
            title={
                'text': "SKU Performance Matrix",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Margin",
            yaxis_title="Year-over-Year Growth",
            showlegend=True,
            legend_title="Premium Status",
            height=450,
            margin=dict(l=20, r=20, t=50, b=50),
            paper_bgcolor=portfolio['color_scheme']['background'],
            plot_bgcolor=portfolio['color_scheme']['background']
        )

        # Update hover template
        fig.update_traces(
            hovertemplate='<b>%{customdata[0]}</b><br>Margin: %{x:.2f}<br>Growth: %{y:.1%}<br>Volume: %{customdata[1]:,}'
        )

        return fig

    # Create brand mix pie chart
    def create_brand_mix(portfolio_key):
        portfolio = portfolios[portfolio_key]

        # Convert brand mix data to DataFrame
        brand_df = pd.DataFrame(portfolio['brand_mix'])

        # Create custom colorscale based on portfolio
        if portfolio_key == 'kuwait':
            colors = px.colors.sequential.Blues[3:]
        else:
            colors = px.colors.sequential.Reds[3:]

        # Create the pie chart
        fig = px.pie(
            brand_df,
            values='value',
            names='name',
            color_discrete_sequence=colors,
            hole=0.4
        )

        # Update layout
        fig.update_layout(
            title={
                'text': "Brand Mix by Volume",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            legend_title="Brand",
            height=400,
            margin=dict(l=20, r=20, t=50, b=50),
            paper_bgcolor=portfolio['color_scheme']['background'],
            plot_bgcolor=portfolio['color_scheme']['background']
        )

        # Update traces
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )

        return fig

    # Create KPI summary table
    def create_kpi_summary(portfolio_key):
        portfolio = portfolios[portfolio_key]

        # Create performance metrics for display
        if portfolio_key == 'kuwait':
            metrics = [
                ["Market Share", portfolio['metrics']['Market Share'], "#2ecc71"],
                ["Average Margin", portfolio['metrics']['Average Margin'], "#2ecc71"],
                ["Premium Mix", portfolio['metrics']['Premium Mix'], "#2ecc71"],
                ["Growth Rate", portfolio['metrics']['Growth Rate'], "#2ecc71"],
                ["Total Volume", portfolio['metrics']['Total Volume'], "black"],
                ["Green SKUs", portfolio['metrics']['Green SKUs'], "#2ecc71"],
                ["Red SKUs", portfolio['metrics']['Red SKUs'], "#2ecc71"]
            ]
        else:
            metrics = [
                ["Market Share", portfolio['metrics']['Market Share'], "#2ecc71"],  # Market share is good
                ["Average Margin", portfolio['metrics']['Average Margin'], "#e67e22"],
                ["Premium Mix", portfolio['metrics']['Premium Mix'], "#e74c3c"],
                ["Growth Rate", portfolio['metrics']['Growth Rate'], "#e74c3c"],
                ["Total Volume", portfolio['metrics']['Total Volume'], "black"],
                ["Green SKUs", portfolio['metrics']['Green SKUs'], "#e74c3c"],
                ["Red SKUs", portfolio['metrics']['Red SKUs'], "#e74c3c"]
            ]

        fig = go.Figure(data=[
            go.Table(
                header=dict(
                    values=["<b>Key Performance Metrics</b>", "<b>Value</b>"],
                    fill_color=portfolio['color_scheme']['primary'],
                    align='left',
                    font=dict(color='white', size=14)
                ),
                cells=dict(
                    values=[
                        [m[0] for m in metrics],
                        [m[1] for m in metrics]
                    ],
                    fill_color=[portfolio['color_scheme']['background']],
                    align='left',
                    font=dict(color=[m[2] for m in metrics], size=14),
                    height=30
                )
            )
        ])

        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor=portfolio['color_scheme']['background']
        )

        return fig

    # Create summary section
    def create_summary_table(portfolio_key):
        portfolio = portfolios[portfolio_key]

        # Define data for summary table
        if portfolio_key == 'kuwait':
            summary_data = [
                ["Total SKUs", portfolio['summary']['totalSKUs'], portfolio['context']['portfolio']],
                ["Portfolio Score", portfolio['summary']['portfolioScore'], portfolio['context']['location']],
                ["Growth Trend", portfolio['summary']['growthTrend'], portfolio['context']['trend']],
                ["Segment Strength", portfolio['summary']['segmentStrength'], portfolio['context']['segment']],
                ["Margin Profile", portfolio['summary']['marginProfile'], portfolio['context']['margin']],
                ["SKU Health", portfolio['summary']['skuHealth'], portfolio['context']['health']]
            ]
        else:
            summary_data = [
                ["Total SKUs", portfolio['summary']['totalSKUs'], portfolio['context']['portfolio']],
                ["Portfolio Score", portfolio['summary']['portfolioScore'], portfolio['context']['location']],
                ["Growth Trend", portfolio['summary']['growthTrend'], portfolio['context']['trend']],
                ["Segment Strength", portfolio['summary']['segmentStrength'], portfolio['context']['segment']],
                ["Margin Profile", portfolio['summary']['marginProfile'], portfolio['context']['margin']],
                ["SKU Health", portfolio['summary']['skuHealth'], portfolio['context']['health']],
                ["Optimization Focus", "Premium Mix", "Increase premium SKU representation"]
            ]

        fig = go.Figure(data=[
            go.Table(
                header=dict(
                    values=["<b>Metric</b>", "<b>Value</b>", "<b>Context</b>"],
                    fill_color='#333333',
                    align='center',
                    font=dict(color='white', size=14)
                ),
                cells=dict(
                    values=[
                        [row[0] for row in summary_data],
                        [row[1] for row in summary_data],
                        [row[2] for row in summary_data]
                    ],
                    fill_color=[
                        ['#222222', '#1a1a1a'] * (len(summary_data) // 2 + 1),
                        ['#222222', '#1a1a1a'] * (len(summary_data) // 2 + 1),
                        ['#222222', '#1a1a1a'] * (len(summary_data) // 2 + 1)
                    ],
                    align='center',
                    font=dict(color='white', size=12),
                    height=40
                )
            )
        ])

        fig.update_layout(
            height=300 if portfolio_key == 'kuwait' else 350,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor=portfolio['color_scheme']['background']
        )

        return fig

    # App layout
    app.layout = html.Div([
        # Header with portfolio selector
        html.Div([
            html.H1("Portfolio Performance Dashboard",
                    style={'textAlign': 'center', 'marginBottom': '10px', 'marginTop': '20px'}),
            html.Div([
                dcc.RadioItems(
                    id='portfolio-selector',
                    options=[
                        {'label': 'Kuwait (High Performer)', 'value': 'kuwait'},
                        {'label': 'Jeju (Requires Optimization)', 'value': 'jeju'}
                    ],
                    value='kuwait',
                    inputStyle={"margin-right": "5px"},
                    labelStyle={'display': 'inline-block', 'margin': '0 15px', 'font-size': '16px'}
                )
            ], style={'textAlign': 'center', 'marginBottom': '20px'})
        ]),

        # Dashboard content - will be updated by callback
        html.Div(id='dashboard-content')
    ])

    # Define callback to update dashboard based on selected portfolio
    @app.callback(
        Output('dashboard-content', 'children'),
        Input('portfolio-selector', 'value')
    )
    def update_dashboard(selected_portfolio):
        portfolio_data = portfolios[selected_portfolio]

        dashboard_content = [
            # Header with title and score
            html.Div([
                dcc.Graph(figure=create_header(selected_portfolio), config={'displayModeBar': False})
            ], style={'marginBottom': '20px'}),

            # First row - Category scores and Segment distribution
            html.Div([
                html.Div([
                    dcc.Graph(figure=create_radar_chart(selected_portfolio), config={'displayModeBar': False})
                ], className='six columns'),
                html.Div([
                    dcc.Graph(figure=create_segment_chart(selected_portfolio), config={'displayModeBar': False})
                ], className='six columns'),
            ], className='row', style={'marginBottom': '20px'}),

            # Second row - SKU Matrix and KPI metrics
            html.Div([
                html.Div([
                    dcc.Graph(figure=create_sku_matrix(selected_portfolio), config={'displayModeBar': False})
                ], className='eight columns'),
                html.Div([
                    dcc.Graph(figure=create_kpi_summary(selected_portfolio), config={'displayModeBar': False})
                ], className='four columns'),
            ], className='row', style={'marginBottom': '20px'}),

            # Third row - Brand mix and Summary
            html.Div([
                html.Div([
                    dcc.Graph(figure=create_brand_mix(selected_portfolio), config={'displayModeBar': False})
                ], className='six columns'),
                html.Div([
                    html.Div([
                        html.H3("Portfolio Performance Summary",
                                style={'textAlign': 'center', 'backgroundColor': '#333333',
                                       'color': 'white', 'padding': '10px', 'borderRadius': '5px'})
                    ]),
                    dcc.Graph(figure=create_summary_table(selected_portfolio), config={'displayModeBar': False})
                ], className='six columns'),
            ], className='row'),

            # Footer
            html.Div([
                html.P(f"Generated on 2025-02-28 | Portfolio Analysis",
                       style={'textAlign': 'center', 'color': '#888888', 'padding': '20px'})
            ])
        ]

        return dashboard_content

    return app


# Create and run the app
if __name__ == '__main__':
    dashboard_app = create_portfolio_dashboard()
    dashboard_app.run_server(debug=True)
