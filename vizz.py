import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output


def create_portfolio_dashboard():
    # Initialize the Dash app
    app = dash.Dash(__name__, title="Portfolio Performance Dashboard")

    # Portfolio data
    portfolios = {
        'Kuwait': {
            'score': 8.26,
            'status': "HIGH PERFORMER",
            'status_color': "#3498db",
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
                'Green SKUs': 5,
                'Red SKUs': 2
            },
            'segments': {
                'Full Flavor': 40,
                'Light': 35,
                'Menthol': 15,
                'Ultra Light': 10
            }
        },
        'Jeju': {
            'score': 1.97,
            'status': "REQUIRES OPTIMIZATION",
            'status_color': "#e74c3c",
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
                'Green SKUs': 0,
                'Red SKUs': 8
            },
            'segments': {
                'Full Flavor': 70,
                'Light': 30,
                'Menthol': 0,
                'Ultra Light': 0
            }
        }
    }

    # Create SKU data
    sku_data = pd.DataFrame([
        # Kuwait SKUs
        {'Portfolio': 'Kuwait', 'SKU': 'MARLBORO', 'Margin': 0.85, 'Growth': 0.18, 'Volume': 700000, 'Premium': True},
        {'Portfolio': 'Kuwait', 'SKU': 'PARLIAMENT', 'Margin': 0.87, 'Growth': 0.15, 'Volume': 650000, 'Premium': True},
        {'Portfolio': 'Kuwait', 'SKU': 'HEETS', 'Margin': 0.82, 'Growth': 0.25, 'Volume': 600000, 'Premium': True},
        {'Portfolio': 'Kuwait', 'SKU': 'L&M', 'Margin': 0.72, 'Growth': 0.08, 'Volume': 550000, 'Premium': False},
        {'Portfolio': 'Kuwait', 'SKU': 'CHESTERFIELD', 'Margin': 0.90, 'Growth': 0.20, 'Volume': 500000,
         'Premium': True},

        # Jeju SKUs
        {'Portfolio': 'Jeju', 'SKU': 'MARLBORO', 'Margin': 0.69, 'Growth': -0.02, 'Volume': 300000, 'Premium': True},
        {'Portfolio': 'Jeju', 'SKU': 'PARLIAMENT', 'Margin': 0.71, 'Growth': 0.01, 'Volume': 250000, 'Premium': True},
        {'Portfolio': 'Jeju', 'SKU': 'L&M', 'Margin': 0.63, 'Growth': -0.05, 'Volume': 350000, 'Premium': False},
        {'Portfolio': 'Jeju', 'SKU': 'BOND', 'Margin': 0.58, 'Growth': -0.12, 'Volume': 200000, 'Premium': False},
        {'Portfolio': 'Jeju', 'SKU': 'LARK', 'Margin': 0.60, 'Growth': -0.09, 'Volume': 300000, 'Premium': False}
    ])

    # Create radar chart function
    def create_radar_chart():
        fig = go.Figure()

        categories = list(portfolios['Kuwait']['category_scores'].keys())
        kuwait_values = list(portfolios['Kuwait']['category_scores'].values())
        jeju_values = list(portfolios['Jeju']['category_scores'].values())

        # Add Kuwait data
        fig.add_trace(go.Scatterpolar(
            r=kuwait_values + [kuwait_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Kuwait',
            line=dict(color='#3498db'),
            fillcolor='rgba(52, 152, 219, 0.2)'
        ))

        # Add Jeju data
        fig.add_trace(go.Scatterpolar(
            r=jeju_values + [jeju_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Jeju',
            line=dict(color='#e74c3c'),
            fillcolor='rgba(231, 76, 60, 0.2)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            showlegend=True,
            title="Category Performance",
            height=400
        )

        return fig

    # Create SKU performance matrix
    def create_sku_matrix():
        fig = px.scatter(
            sku_data,
            x='Margin',
            y='Growth',
            size='Volume',
            color='Portfolio',
            symbol='Premium',
            hover_name='SKU',
            size_max=40,
            labels={'Growth': 'YOY Growth', 'Margin': 'Margin'},
            color_discrete_map={'Kuwait': '#3498db', 'Jeju': '#e74c3c'}
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

        fig.update_layout(
            title="SKU Performance Matrix",
            height=450
        )

        return fig

    # Create segment distribution chart
    def create_segment_distribution():
        # Combine segment data
        segments = []
        for portfolio, data in portfolios.items():
            for segment, value in data['segments'].items():
                segments.append({
                    'Portfolio': portfolio,
                    'Segment': segment,
                    'Percentage': value
                })

        segment_df = pd.DataFrame(segments)

        fig = px.bar(
            segment_df,
            x='Percentage',
            y='Segment',
            color='Portfolio',
            barmode='group',
            text='Percentage',
            labels={'Percentage': 'Share of Portfolio (%)'},
            color_discrete_map={'Kuwait': '#3498db', 'Jeju': '#e74c3c'},
            orientation='h'
        )

        fig.update_traces(texttemplate='%{text}%', textposition='outside')

        fig.update_layout(
            title="Segment Distribution",
            height=350
        )

        return fig

    # Create metrics comparison table
    def create_metrics_comparison():
        # Convert metrics to DataFrame for easier display
        metrics_list = []
        for portfolio, data in portfolios.items():
            for metric, value in data['metrics'].items():
                metrics_list.append({
                    'Portfolio': portfolio,
                    'Metric': metric,
                    'Value': value
                })

        metrics_df = pd.DataFrame(metrics_list)

        # Determine colors based on metric type and value
        def determine_color(row):
            metric = row['Metric']
            value = row['Value']
            portfolio = row['Portfolio']

            if metric in ['Growth Rate']:
                if isinstance(value, str) and '+' in value:
                    return 'green'
                elif isinstance(value, str) and '-' in value:
                    return 'red'
                elif isinstance(value, (int, float)) and value > 0:
                    return 'green'
                else:
                    return 'red'
            elif metric in ['Green SKUs']:
                return 'green'
            elif metric in ['Red SKUs']:
                if value > 5:
                    return 'red'
                elif value > 2:
                    return 'orange'
                else:
                    return 'green'
            elif metric in ['Average Margin', 'Premium Mix']:
                if portfolio == 'Kuwait':
                    return 'green'
                else:
                    return 'orange'
            else:
                return 'black'

        metrics_df['Color'] = metrics_df.apply(determine_color, axis=1)

        # Create figure
        fig = go.Figure(data=[
            go.Table(
                header=dict(
                    values=['Metric', 'Kuwait', 'Jeju'],
                    fill_color='#f0f0f0',
                    align='left',
                    font=dict(size=14)
                ),
                cells=dict(
                    values=[
                        metrics_df[metrics_df['Portfolio'] == 'Kuwait']['Metric'].unique(),
                        metrics_df[metrics_df['Portfolio'] == 'Kuwait']['Value'],
                        metrics_df[metrics_df['Portfolio'] == 'Jeju']['Value']
                    ],
                    align='left',
                    font=dict(size=13),
                    height=30
                )
            )
        ])

        fig.update_layout(
            title="Key Performance Metrics",
            height=350,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        return fig

    # Create score cards
    def create_score_card(portfolio):
        data = portfolios[portfolio]

        fig = go.Figure()

        # Add score text
        fig.add_trace(go.Indicator(
            mode="number",
            value=data['score'],
            number={"font": {"size": 48, "color": data['status_color']}},
            title={"text": f"<b>{portfolio}</b><br><span style='font-size:0.8em;'>{data['status']}</span>",
                   "font": {"size": 16}},
            domain={"row": 0, "column": 0}
        ))

        fig.update_layout(
            height=160,
            margin=dict(l=20, r=20, t=30, b=30),
            paper_bgcolor="#f8f9fa",
            plot_bgcolor="#f8f9fa"
        )

        return fig

    # App layout
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1("Portfolio Performance Dashboard", className="text-center p-4"),
        ], className="bg-dark text-white mb-4"),

        # Score Cards
        html.Div([
            html.Div([
                dcc.Graph(figure=create_score_card('Kuwait'), config={'displayModeBar': False})
            ], className="col-md-6"),
            html.Div([
                dcc.Graph(figure=create_score_card('Jeju'), config={'displayModeBar': False})
            ], className="col-md-6")
        ], className="row mb-4"),

        # First row of charts
        html.Div([
            html.Div([
                dcc.Graph(figure=create_radar_chart())
            ], className="col-md-6"),
            html.Div([
                dcc.Graph(figure=create_segment_distribution())
            ], className="col-md-6")
        ], className="row mb-4"),

        # Second row of charts
        html.Div([
            html.Div([
                dcc.Graph(figure=create_sku_matrix())
            ], className="col-md-7"),
            html.Div([
                dcc.Graph(figure=create_metrics_comparison())
            ], className="col-md-5")
        ], className="row mb-4"),

        # Footer
        html.Div([
            html.P("Generated on 2025-02-28 | Portfolio Analysis", className="text-center text-muted")
        ], className="footer mt-4")
    ], className="container-fluid")

    return app


# Create and run the app
if __name__ == '__main__':
    dashboard_app = create_portfolio_dashboard()
    dashboard_app.run_server(debug=True)