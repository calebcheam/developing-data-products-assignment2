import dash
from dash import dash_table
from dash import Dash, html, dcc, Input, Output, State
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import gensim
from gensim import corpora
from sklearn.decomposition import PCA
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
import ast
import networkx as nx
from fuzzywuzzy import fuzz



DEFAULT_IMAGE_URL = 'https://logowik.com/content/uploads/images/ntu-nanyang-technological-university9496.logowik.com.webp'


# Sample Data
# sample_data = {
#     'Full Name': ['A S Madhukumar', 'Alexei Sourin', 'Anupam Chattopadhyay'],
#     'Email': ['asmadhukumar@ntu.edu.sg', 'assourin@ntu.edu.sg', 'anupam@ntu.edu.sg'],
#     'DR_NTU_URL': ['https://dr.ntu.edu.sg/cris/rp/rp00083', 'https://dr.ntu.edu.sg/cris/rp/rp00274', 'https://dr.ntu.edu.sg/cris/rp/rp01076'],
#     'Website_URL': ['http://www3.ntu.edu.sg/home/asmadhukumar/', 'http://www3.ntu.edu.sg/home/assourin/', 'https://scholar.google.co.in/citations?user=TI...'],
#     'DBLP_URL': ['https://dblp.org/pid/66/549.html', 'https://dblp.org/pid/15/3108.html', 'https://dblp.org/pid/99/4535.html'],
#     'Citations(All)': [2907, 2939, 6226],
#     'Biography': ['Bio 1', 'Bio 2', 'Bio 3'],
#     'Research': [
#         'Artificial intelligence and machine learning applications...',
#         'Shape modeling, multi-modal interaction, new user interfaces...',
#         'Computing ArchitectureDesign AutomationSecurity...'
#     ],
#     'Grants': ['Grant 1', 'Grant 2', 'Grant 3'],
#     'Keywords': ['Keyword 1', 'Keyword 2', 'Keyword 3'],
# }
metrics_df = pd.read_csv('metrics.csv')

jaccard_df = pd.read_csv("pairwise_jaccard.csv")

df = pd.read_csv("final_df.csv")

df[['Education', 'Awards']] = df[['Education', 'Awards']].astype(str)

original_df = pd.read_csv('original_data.csv')
ntu_profs = original_df['Full Name'].tolist()

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def network_graph(collaboration_df:pd.DataFrame,target_author:str):

    G = nx.Graph()

    for index, row in collaboration_df.iterrows():
        source = row['Source']
        target = row['Target']
        affiliation = row['Affiliation']

        if G.has_edge(source, target):
            G[source][target]['weight'] += 1
        else:
            G.add_edge(source, target, weight=1, affiliation=affiliation)

    # Extract subgraph for the target author
    subgraph_nodes = [neighbor for neighbor in G.neighbors(target_author)] + [target_author]
    H = G.subgraph(subgraph_nodes)

    # Positioning of nodes
    pos = nx.spring_layout(H)

    # Edge traces
    edge_traces = []
    for edge in H.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        width = H[edge[0]][edge[1]]['weight']
        color = 'blue' if H[edge[0]][edge[1]]['affiliation'] == 'NTU' else 'green'

        edge_trace = go.Scatter(
            x=[x0, x1], y=[y0, y1],
            line=dict(width=width, color=color),
            hoverinfo='none',
            mode='lines')
        edge_traces.append(edge_trace)

    # Node trace
    node_x = [pos[node][0] for node in H.nodes()]
    node_y = [pos[node][1] for node in H.nodes()]

    node_sizes = [15 if node == target_author else 10 + 5*H[target_author][node]['weight'] for node in H.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            colorscale='YlGnBu',
            size=node_sizes,
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node in H.nodes():
        collaborations_with_target = H[target_author][node]['weight'] if node != target_author else ''
        node_text.append(f"{node}<br>Collaborations with {target_author}: {collaborations_with_target}")

    node_trace.marker.color = [len(list(H.neighbors(node))) for node in H.nodes()]
    node_trace.text = node_text

    # Create the figure
    fig = go.Figure(data=edge_traces + [node_trace],
                 layout=go.Layout(
                    title='Collaboration Network of ' + target_author,
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

def compute_avg_score_plot(input_df:pd.DataFrame):
    df = pd.DataFrame()
    for __, row in input_df.iterrows():
        new_df = pd.DataFrame(ast.literal_eval(row["publications_detailed_data"]))
        df = pd.concat([df, new_df], axis=0)
    
    df['venue_score'] = pd.to_numeric(df['venue_score'], errors='coerce')
    df = df[df['year'] > 1994]
    avg_scores_per_year = df.groupby('year')['venue_score'].mean().reset_index()

    # Plotting using Plotly
    fig = px.line(avg_scores_per_year, x='year', y='venue_score', title='Average Venue Score Across Years for SCSE')
    fig.update_layout(xaxis_title='Year', yaxis_title='Average Venue Score')
    
    return fig

def compute_avg_score_by_subtopic(input_df):
    df = pd.DataFrame()
    for index, row in input_df.iterrows():
        new_df = pd.DataFrame(ast.literal_eval(row["publications_detailed_data"]))
        df = pd.concat([df, new_df], axis=0)
    
    df['venue_score'] = pd.to_numeric(df['venue_score'], errors='coerce')
    
    avg_scores_per_subtopic = df.groupby('subtopic')['venue_score'].mean().reset_index().dropna().sort_values(by='venue_score', ascending=True)

    # Plotting using Plotly
    fig = px.bar(avg_scores_per_subtopic, y='subtopic', x='venue_score', title='Average Venue Score by Subtopic for SCSE', color_discrete_sequence=['#636EFA'], orientation='h')
    fig.update_layout(yaxis_title='Subtopic', xaxis_title='Average Venue Score', height=800)
    
    # Ensure axis alignment
    fig.update_yaxes(categoryorder='total ascending')
    
    return fig


def visualize_top_venues(input_df, top_n=20):
    df = pd.DataFrame()
    for index, row in input_df.iterrows():
        new_df = pd.DataFrame(ast.literal_eval(row["publications_detailed_data"]))
        df = pd.concat([df, new_df], axis=0)
    
    df['venue_score'] = pd.to_numeric(df['venue_score'], errors='coerce')
    
    # Count the number of publications in each venue
    venue_counts = df['venue_name'].value_counts().reset_index()
    venue_counts.columns = ['venue_name', 'count']
    
    # Compute average score for each venue
    avg_scores = df.groupby('venue_name')['venue_score'].mean().reset_index()
    
    # Merge the two dataframes
    merged = pd.merge(venue_counts, avg_scores, on='venue_name')
    
    # Filter to top_n venues
    merged = merged.sort_values(by='count', ascending=False).head(top_n)
    
    # Plotting
    fig = px.bar(merged,
                 y='venue_name', x='count',
                 color='venue_score',
                 labels={'venue_name': 'Venue', 'count': 'Number of Publications'},
                 title=f'Top {top_n} Venues by Publication Count for SCSE',
                 color_continuous_scale=px.colors.sequential.Plasma,
                 )
    
    fig.update_layout(showlegend=False, yaxis_showticklabels=False)
    return fig
    
    
def plot_expertise_distribution(input_df):
    df = input_df.copy()

    unique_keywords = set(kw for kws in df['keywords'].dropna() for kw in str(kws).split(', '))
    df['keywords'] = df['keywords'].apply(lambda x: str(x).split(', ') if x and not pd.isna(x) else [])

    for keyword in unique_keywords:
        df[keyword] = df['keywords'].apply(lambda x: 1 if keyword in x else 0)

    df = df.drop(columns = input_df.drop(columns=["Full Name"]).columns.tolist())

    sum_expertise = df.sum().reset_index()
    sum_expertise.columns = ['Expertise Area', 'Count']
    sum_expertise = sum_expertise[sum_expertise['Expertise Area'] != 'Full Name']

    sum_expertise = sum_expertise[sum_expertise['Expertise Area'] != 'Computer Science and Engineering']

    fig = px.pie(sum_expertise, 
                 names='Expertise Area', 
                 values='Count', 
                 hole=0.3)  

    fig.update_layout(annotations=[dict(text='SCSE Expertise', x=0.5, y=0.5, font_size=15, showarrow=False)],
                    #   margin=dict(t=100, b=0, l=0, r=0), 
                    #   title_font=dict(size=24)  
                     )
    
    fig.update_layout(width=1100, height=1100)


    return fig


def plot_subtopic_distribution(input_df):
    df = pd.DataFrame()
    for index, row in input_df.iterrows():
        new_df = pd.DataFrame(ast.literal_eval(row["publications_detailed_data"]))
        df = pd.concat([df, new_df], axis=0)
    
    subtopic_counts = df['subtopic'].value_counts().reset_index()
    subtopic_counts.columns = ['Subtopic', 'Count']

    fig = px.pie(subtopic_counts, 
                 names='Subtopic', 
                 values='Count', 
                 hole=0.3,
                 title="Distribution of Subtopics across SCSE Publications")
    
    fig.update_layout(annotations=[dict(text='SCSE Subtopics', x=0.5, y=0.5, font_size=15, showarrow=False)])
    
    return fig

def clean_text(text):
    text = str(text)
    text = re.sub(r'\W|\d', ' ', text)  # Remove special chars and digits
    words = text.lower().split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Remove stopwords and lemmatize
    return words

def get_topic_modelling_figure(df):
    df['clean_research'] = df['research'].apply(clean_text)
    dictionary = corpora.Dictionary(df['clean_research'])
    corpus = [dictionary.doc2bow(text) for text in df['clean_research']]
    lda_model = gensim.models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=30)

    num_topics = lda_model.num_topics
    
    top_words_per_topic = []
    for t in range(num_topics):
        top_words = lda_model.show_topic(t, topn=3)
        top_words_per_topic.append(", ".join([word for word, _ in top_words]))
    
    dominant_topic = []
    doc_topic_dist = []

    for i in range(len(df)):
        topic_probs = np.zeros(num_topics)
        for topic, prob in lda_model[corpus[i]]:
            topic_probs[topic] = prob
        doc_topic_dist.append(topic_probs)
        dominant_topic.append(np.argmax(topic_probs))

    # Convert to NumPy array
    doc_topic_dist = np.array(doc_topic_dist)

    # PCA and Plotting
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(doc_topic_dist)
    
    plot_df = pd.DataFrame({
        'pc1': pca_result[:,0], 
        'pc2': pca_result[:,1], 
        'topic': dominant_topic,
        'Full Name': df['Full Name'],
        'Top Words': [top_words_per_topic[t] for t in dominant_topic]  # Map dominant_topic to top words
    })

    fig = px.scatter(
        plot_df, 
        x='pc1', 
        y='pc2', 
        color='topic', 
        title='2D Visualization of Topics',
        hover_data=['Full Name', 'Top Words']  
    )
    return fig

def get_line_plot_prof(df):
    fig = px.bar(df, x='Year', y='Citations', title='Citations per Year')
    fig.add_trace(px.line(df, x='Year', y='Citations').data[0])

    # Customize the layout
    fig.update_layout(
        title='Yearly Citations',
        xaxis_title='Year',
        yaxis_title='Citations',
        showlegend=False,
        template='plotly'  # or 'plotly', 'plotly_white', etc.
    )
    return fig

def get_publications_per_year(df):
    pub_per_year = df.groupby('year').size().reset_index(name='publications')
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=pub_per_year['year'],
        y=pub_per_year['publications'],
        name='Publications',
        marker_color='#636EFA'  # Blue color
    ))

    fig.add_trace(go.Scatter(
        x=pub_per_year['year'],
        y=pub_per_year['publications'],
        mode='lines+markers',
        name='Trend',
        line=dict(color='red', width=2),
        marker=dict(size=6, color='red')
    ))

    fig.update_layout(
        title='Yearly Publications',
        xaxis_title='Year',
        yaxis_title='Number of Publications',
        showlegend=True,
        template='plotly'  # or 'plotly', 'plotly_white', etc.
    )
    return fig
    

# App Layout
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container(
    [
        dbc.Row(dbc.Col(html.H1("SCSE Professor Dashboard", style={'color': '#00509E'}), width=12), className="mb-4 text-center"),
        dbc.Tabs([
            dbc.Tab(label='Professor Search', children=[
                dbc.InputGroup(
                    [
                        dbc.Input(id='search-box', placeholder='Search professor...', type='text'),
                        dbc.Button('Search', id='search-btn', color='primary'),
                    ],
                    className='mb-3'
                ),
                
                dbc.Card(
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Img(id='prof-pic', src='', style={'width': '150px', 'height': '150px'}),
                                        width=3,
                                        className='mb-3 mb-md-0',
                                    ),
                                    dbc.Col(
                                        [
                                            html.H2(id='prof-name', children='', style={'color': '#00509E'}),
                                            html.Details([
                                                html.Summary("Biography", style={'color': '#DAA520'}),
                                                html.P(id='prof-bio', children='', style={'color': '#555555'}),
                                            ]),
                                            html.Details([
                                                html.Summary("Research Interests", style={'color': '#DAA520'}),
                                                html.P(id='prof-interests', children='', style={'color': '#555555'}),
                                            ]),
                                            dbc.ButtonGroup(
                                                [
                                                    dbc.Button('DR_NTU Profile', id='prof-drntu-link', href='', target='_blank', color='link', style={'display': 'none'}),
                                                    dbc.Button('Website', id='prof-website-link', href='', target='_blank', color='link', style={'display': 'none'}),
                                                    dbc.Button('DBLP Profile', id='prof-dblp-link', href='', target='_blank', color='link', style={'display': 'none'}),
                                                ],
                                                className='mb-3',
                                            ),
                                        ],
                                        width=8,
                                    ),
                                ],
                                align='start',
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H4('Details:', style={'color': '#00509E'}),
                                            html.Details([
                                                html.Summary('Education', style={'color': '#DAA520'}),
                                                dbc.ListGroup(id='prof-education', flush=True)
                                            ]),
                                            html.Details([
                                                html.Summary('Work Experience', style={'color': '#DAA520'}),
                                                dbc.ListGroup(id='prof-work-experience', flush=True)
                                            ]),
                                            html.Details([
                                                html.Summary('Awards', style={'color': '#DAA520'}),
                                                dbc.ListGroup(id='prof-awards', flush=True)
                                            ]),
                                            html.Details([
                                                html.Summary('Others', style={'color': '#DAA520'}),
                                                dbc.ListGroup(id='prof-others', flush=True)
                                            ]),
                                        ],
                                        width=12,
                                    )
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H4('Grants:', style={'color': '#00509E'}),
                                            html.Ul(id='prof-grants', children=[]),
                                        ],
                                        width=5,
                                    ),
                                    dbc.Col(
                                        [
                                            html.H4('Keywords:', style={'color': '#00509E'}),
                                            html.Ul(id='prof-keywords', children=[]),
                                        ],
                                        width=5,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    className='mb-3',
                    style={'background-color': '#f8f9fa'},
                ),
                dbc.Card(
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H4('Works', className='card-title', style={'color': '#00509E'}),
                                dcc.Graph(id='prof-line-plot', figure={}, style={'display': 'none'}),
                                dcc.Graph(id='prof-line-plot-publications', figure={}),
                            ], width=7),
                            dbc.Col([
                                html.H4('Collaboration Stats', className='card-title', style={'color': '#00509E'}),
                                html.Details([
                                                html.Summary("Non-NTU Co-authors:", style={'color': '#DAA520'}),
                                                html.P(id='non-ntu-co-authors', children='', className='card-text', style={'color': '#555555'}),
                                            ]),
                                html.Details([
                                                html.Summary("NTU Co-authors:", style={'color': '#DAA520'}),
                                                html.P(id='ntu-co-authors', children='' , className = 'card-text', style={'color': '#555555'}),
                                            ]),
                                dcc.Graph(id='prof-network', figure={}) 
                            ], width=5)
                        ])
                    ]),
                className='mb-3'
                ),
                dbc.Card(
                        dbc.CardBody([
                            html.H4('Research Sub Topics', className='research-topics-title', style={'color': '#00509E'}),
                            dbc.Row([
                                dbc.Col([
                                    dash_table.DataTable(
                                        id='table',
                                        style_header={
                                            'backgroundColor': '#00509E',
                                            'color': 'white',
                                            'fontWeight': 'bold'
                                        },
                                        style_cell={
                                            'backgroundColor': '#f4f4f4',  # Light gray
                                            'color': '#333333',  # Darker text for better readability
                                            'border': '1px solid white'
                                        },
                                    )
                                ], width=5),
                                dbc.Col([
                                    dcc.Graph(id='prof-subtopic-piechart', figure={}) 
                                ], width=7)  
                                
                            ])  
                        ]),  
                    ),  
                dbc.Card(
                    dbc.CardBody([
                        html.H4('Similar Researchers', className='similar-professors-title', style={'color': '#00509E'}),
                            dcc.Graph(id='similar-researchers', figure={}, style={'color': '#555555'}),  # Replace the html.Ul with dcc.Graph to display the dot plot
                        ]),
                    ),
                dbc.Card(
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H4('Venue Statistics', className='venue-card-title', style={'color': '#00509E'}),
                                dcc.Graph(id='venue-histogram', figure={}),
                            ])
                        ])
                    ]),
                    className='mb-3',
                    style={'background-color': '#f8f9fa'},
                ),
            ]),
            dbc.Tab(label='Topic Modeling', children=[
                dcc.Graph(id='topic-plot', figure=get_topic_modelling_figure(df))  # Add the topic modeling plot here
            ]),
            dbc.Tab(
                label="SCSE Faculty Infographics",
                children=[
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4('Average Score Plot', className='card-title', style={'color': '#00509E'}),
                            dcc.Graph(id='avg-score-plot', figure=compute_avg_score_plot(df)),
                            html.H4('Top Venues', className='card-title', style={'color': '#00509E'}),
                            dcc.Graph(id='top-venues', figure=visualize_top_venues(df))
                        ]),
                    ),
                    className='mb-4',
                    width=6
                    
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4('Average Score by Subtopic', className='card-title', style={'color': '#00509E'}),
                            dcc.Graph(id='avg-score-subtopic', figure=compute_avg_score_by_subtopic(df))
                        ]),
                    ),
                    className='mb-4',
                    width=6
                ),
            ]),
            dbc.Row([
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody([
                                html.H4('Expertise Distribution', className='card-title', style={'color': '#00509E'}),
                                dcc.Graph(id='expertise-distribution', figure=plot_expertise_distribution(df))
                            ]),
                            className='mb-4'
                        ),
                        dbc.Card(
                            dbc.CardBody([
                                html.H4('Professor Metrics', className='card-title', style={'color': '#00509E'}),
                                dcc.Dropdown(
                                    id='prof-dropdown',
                                    options=[{'label': name, 'value': name} for name in metrics_df['Full Name']],
                                    value=[],
                                    multi=True
                                ),
                                dcc.Graph(id='radar-chart')
                            ]),
                            className='mb-4'
                        )
                    ],
                    width=10
                )
            ])
        ],
)

        ]),
    ],
    fluid=True,
)

# Callbacks
@app.callback(
    [Output('prof-pic', 'src'),
     Output('prof-name', 'children'),
     Output('prof-bio', 'children'),
     Output('prof-interests', 'children'),
     Output('prof-grants', 'children'),
     Output('prof-keywords', 'children'),
     Output('prof-drntu-link', 'href'),
     Output('prof-website-link', 'href'),
     Output('prof-dblp-link', 'href'),
     Output('prof-drntu-link', 'style'),
     Output('prof-website-link', 'style'),
     Output('prof-dblp-link', 'style'),
     Output('prof-line-plot', 'figure'),
     Output('prof-line-plot', 'style'),
     Output('non-ntu-co-authors', 'children'),
     Output('ntu-co-authors', 'children'),
     Output('prof-network', 'figure'),
     Output('similar-researchers', 'figure'),
     Output('prof-education', 'children'),
     Output('prof-work-experience', 'children'),
     Output('prof-awards', 'children'),
     Output('prof-others', 'children'),
     Output('table', 'data'),
     Output('table', 'columns'),
     Output('prof-subtopic-piechart', 'figure'),
     Output('prof-line-plot-publications', 'figure'),
     Output('venue-histogram', 'figure'),
    ],
    [Input('search-btn', 'n_clicks')],
    [State('search-box', 'value')]
)
def update_prof_data(n_clicks, search_value):
    if n_clicks is None:
       return 'https://logowik.com/content/uploads/images/ntu-nanyang-technological-university9496.logowik.com.webp','', '', '', '', '', '', '', '', {'display': 'none'}, {'display': 'none'}, {'display': 'none'},{}, {'display': 'none'}, 'Non-NTU Co-authors: 0', 'NTU Co-authors: 0', {},{},[], [], [], [] , [], [], {},{},{}
    
    filtered_df = df[df['Full Name'].str.contains(search_value, case=False, na=False)]

    if not filtered_df.empty:
        prof = filtered_df.iloc[0]
        
        non_ntu_co_authors = prof["Non_Ntu_Affiliated_Coauthors"]
        ntu_co_authors = prof["NTU_Affiliated_Coauthors"]
        affiliations:set = ast.literal_eval(prof["non_ntu_affliations"])
        affiliations.remove('')
        # affiliations_list_items = [html.Li(affiliation) for affiliation in affiliations] if len(affiliations) != 0 else []
        
        image = prof['image_url'] 
        citations_time_series = prof["citation_data"]
        citations_data = pd.DataFrame(ast.literal_eval(citations_time_series))
        
        line_plot = get_line_plot_prof(citations_data)

        
        #subtopics publications things
        subtopics_df = pd.DataFrame(ast.literal_eval(prof['publications_detailed_data']))
        
        collaboration_data = []

        def is_ntu_affiliated(author, ntu_prof_list):
            for prof in ntu_prof_list:
                if fuzz.ratio(author, prof) > 85:  # You can adjust this threshold as needed
                    return True
            return False

        for index, row in subtopics_df.iterrows():
            for author in row['authors']:
                if fuzz.ratio(author, search_value) < 85:  # Not the target author
                    affiliation = 'NTU' if is_ntu_affiliated(author, ntu_profs) else 'Outside NTU'
                    collaboration_data.append([search_value, author, affiliation])

        collaboration_df = pd.DataFrame(collaboration_data, columns=['Source', 'Target', 'Affiliation'])
        
        network_fig = network_graph(collaboration_df,search_value)
        
        subtopics_df['venue_score'] = pd.to_numeric(subtopics_df['venue_score'], errors='coerce')
        
        publications_line_plot = get_publications_per_year(subtopics_df)

        # Filter out rows where 'venue_score' is NaN
        venue_score_filtered = subtopics_df.dropna(subset=['venue_score'])

        # Compute the average score
        avg_score = venue_score_filtered['venue_score'].mean()
        
        fig_hist_score  = px.histogram(venue_score_filtered, x='venue_score', title='Distribution of Venue Scores', nbins=20, color_discrete_sequence=['#636EFA'])
        fig_hist_score.update_layout(bargap=0.1, xaxis_title='Venue Score', yaxis_title='Count')

        fig_hist_score.add_trace(
            go.Scatter(
                x=[avg_score, avg_score],
                y=[0, venue_score_filtered['venue_score'].value_counts().max()],
                mode="lines",
                line=go.scatter.Line(color="red"),
                showlegend=False
            )
        )
        
        fig_hist_score.add_annotation(
            x=avg_score,
            y=venue_score_filtered['venue_score'].value_counts().max(),
            xref="x",
            yref="y",
            text=f"Average Score: {avg_score:.2f}",
            showarrow=True,
            arrowhead=4,
            ax=0,
            ay=-40
        )
        
        # venue_score_filtered = subtopics_df.dropna(subset=['venue_score'])
        # fig_hist_score  = px.histogram(venue_score_filtered, x='venue_score', title='Distribution of Venue Scores', nbins=20, color_discrete_sequence=['#636EFA'])  # Using a blue color
        # fig_hist_score.update_layout(bargap=0.1, xaxis_title='Venue Score', yaxis_title='Count')
        
        subtopic_counts = subtopics_df['subtopic'].value_counts()
        fig_pie = px.pie(subtopic_counts, values=subtopic_counts.values, names=subtopic_counts.index, title='Distribution of Subtopics', hole=0.3)
        fig_pie.update_traces(textinfo='percent+label', pull=[0.01 for _ in subtopic_counts.index], marker=dict(colors=px.colors.qualitative.Pastel))  # Using Plasma color scale
        fig_pie.update_layout(legend_title_text='Subtopics')

        subtopic_counts = subtopics_df.groupby(['year', 'subtopic']).size().reset_index(name='counts')
        top_subtopics = subtopic_counts.sort_values(by=['year', 'counts'], ascending=[True, False]).drop_duplicates(subset='year')
        
        top_subtopics = top_subtopics.drop(columns=['counts'], errors='ignore')

        # Rename the subtopic column
        top_subtopics = top_subtopics.rename(columns={'subtopic': 'Most Frequent Subtopic'})

        # Sort by year in descending order
        top_subtopics = top_subtopics.sort_values(by='year', ascending=False)

        subtopic_columns = [{"name": i, "id": i} for i in top_subtopics.columns]
        subtopic_data = top_subtopics.to_dict('records')
        
        grants = prof['grants'].split(', ')
        keywords = prof['keywords'].split(', ')

        grants_badges = [dbc.Badge(grant, color="primary", className="mr-1") for grant in grants]
        keywords_badges = [dbc.Badge(keyword, color="secondary", className="mr-1") for keyword in keywords]
        
        similar_profs_df = jaccard_df[
            jaccard_df['person 1'] == search_value
        ].nlargest(10, 'jaccard similarity')
        
        def create_bar_plot(similar_profs_df):
            # Sort the dataframe by Jaccard similarity
            sorted_df = similar_profs_df.sort_values(by='jaccard similarity', ascending=True)            
    
            # Create a color scale based on Jaccard similarity
            colors = ['#d3d3d3' if val < sorted_df['jaccard similarity'].mean() else '#4682B4' for val in sorted_df['jaccard similarity']]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=sorted_df['jaccard similarity'],
                y=sorted_df['person 2'],
                orientation='h',  # horizontal bar plot
                marker=dict(color=colors)
            ))
            fig.update_layout(title='Top 10 Similar Professors', xaxis_title='Jaccard Similarity', yaxis_title='Professor Name')
            return fig
        

        similar_profs_fig = create_bar_plot(similar_profs_df)
        education_list = [dbc.ListGroupItem(edu) for edu in ast.literal_eval(prof['Education'])]
        work_experience_list = [dbc.ListGroupItem(exp) for exp in ast.literal_eval(prof['Work Experience'])]
        awards_list = [dbc.ListGroupItem(award) for award in ast.literal_eval(prof['Awards'])]
        others_list = [dbc.ListGroupItem(other) for other in ast.literal_eval(prof['Others'])]
        
        return (
            image,
            prof['Full Name'],
            prof['biography'],
            f"Research Interests: {prof['research']}",
            grants_badges,
            keywords_badges,
            prof['DR_NTU_URL'],
            prof['Website_URL'],
            prof['DBLP_URL'],
            {'display': 'block'},
            {'display': 'block'},
            {'display': 'block'},
            line_plot,
            {'display': 'block'},
            f'Non-NTU Co-authors: {non_ntu_co_authors}',
            f'NTU Co-authors: {ntu_co_authors}',
            network_fig,
            similar_profs_fig,
            education_list,
            work_experience_list,
            awards_list,
            others_list,
            subtopic_data,
            subtopic_columns,
            fig_pie,
            publications_line_plot,
            fig_hist_score,
        )
    else:
       return 'https://logowik.com/content/uploads/images/ntu-nanyang-technological-university9496.logowik.com.webp','', '', '', '', '', '', '', '', {'display': 'none'}, {'display': 'none'}, {'display': 'none'},{}, {'display': 'none'}, 'Non-NTU Co-authors: 0', 'NTU Co-authors: 0', {},{},[], [], [], [] , [], [], {},{},{}



@app.callback(
    Output('radar-chart', 'figure'),
    [Input('prof-dropdown', 'value')]
)
def update_radar_chart(selected_profs):
    categories = ['citation_publications_ratio', 'avg_venue_score', 'diversification_score']
    fig = go.Figure()
    filtered_df = metrics_df[metrics_df['Full Name'].isin(selected_profs)]
    for index, row in filtered_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['citation_publications_ratio'], row['avg_venue_score'], row['diversification_score']],
            theta=categories,
            fill='toself',
            name=row['Full Name']
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )
    return fig
# Run App
if __name__ == '__main__':
    app.run_server(debug=True)