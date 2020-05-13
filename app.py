import os
from textwrap import dedent

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#from dash.dependencies import Input, Output, State

#import dash_reusable_components as drc

import pandas as pd
import flask

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

server = flask.Flask(__name__)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,server=server)

RANDOM_STATE = 718

app = dash.Dash(__name__)
server = app.server

# Custom Script for Heroku
if 'DYNO' in os.environ:
    app.scripts.append_script({
        'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'
    })

url = "https://raw.githubusercontent.com/sigmasigmaiota/DATA608F/master/SADCQ_2010s.csv"
    
ysrb = pd.read_csv(url, sep=',')

# Discordance in sexual identity and orientation and experiences

ysrb.loc[(ysrb['sex'] >= 1) | (ysrb['sexid'] >= 1) | (ysrb['sexpart2'] >= 1) | ((ysrb['q66'] >= 1)  | (ysrb['sexpart2'] >= 1)), 'discord'] = 0
# if male sex and hetero sexid and sex with females and males
ysrb.loc[(ysrb['sex'] == 2) & (ysrb['sexid'] == 1) & (ysrb['q66'] >= 3), 'discord'] = 1
# if female sex and hetero sexid and sex with females and males
ysrb.loc[(ysrb['sex'] == 1) & (ysrb['sexid'] == 1) & ((ysrb['q66'] == 2) | (ysrb['q66'] == 4)), 'discord'] = 1
# if hetero sexid and sex with females and males
ysrb.loc[(ysrb['sexid'] == 1) & (ysrb['q66'] == 4), 'discord'] = 1
# if gay sexid or bi sexid and sex with opposite sex only
ysrb.loc[(ysrb['sexid2'] == 2) & (ysrb['sexpart2'] == 2), 'discord'] = 1
# if hetero sexid and sex with same sex only or both sexes
ysrb.loc[(ysrb['sexid2'] == 1) & (ysrb['sexpart2'] == 3), 'discord'] = 1
# if hetero q67 and sex with same sex only or both sexes
ysrb.loc[(ysrb['q67'] == 1) & (ysrb['sexid'] >= 2), 'discord'] = 1
# if hetero sexid and sex with same sex only or both sexes q67
ysrb.loc[(ysrb['q67'] >= 2) & (ysrb['sexid'] == 1), 'discord'] = 1

# bullying
ysrb.loc[(ysrb['q23'] >= 1) | (ysrb['q24'] >= 1), 'bullied'] = 0
ysrb.loc[(ysrb['q23'] == 1) | (ysrb['q24'] == 1), 'bullied'] = 1

# suicidal ideation
ysrb.loc[(ysrb['q26'] >= 1) | (ysrb['q28'] >= 1) | (ysrb['q29'] >= 1), 's_idea'] = 0
ysrb.loc[(ysrb['q26'] == 1) | (ysrb['q28'] >= 2) | (ysrb['q29'] >= 2), 's_idea'] = 1

# Experiences with alcohol

#ysrb['alc30days'] = 0
ysrb.loc[(ysrb['q42'] >= 1), 'alc30days'] = 0 
ysrb.loc[(ysrb['q42'] >= 2), 'alc30days'] = 1

#############################################################################
# Experiences with marijuana, first age

ysrb.loc[(ysrb['q47'] >= 1), 'mjage'] = 0 
ysrb.loc[(ysrb['q47'] >= 2), 'mjage'] = 1

#############################################################################
# Experiences with marijuana, 30 day use

ysrb.loc[(ysrb['q48'] >= 1), 'mj30d'] = 0 
ysrb.loc[(ysrb['q48'] >= 2), 'mj30d'] = 1

#############################################################################
# Experiences with cocaine, ever

ysrb.loc[(ysrb['q49'] >= 1), 'coc'] = 0 
ysrb.loc[(ysrb['q49'] >= 2), 'coc'] = 1

#############################################################################
# Experiences with heroin, ever

ysrb.loc[(ysrb['q51'] >= 1), 'her'] = 0 
ysrb.loc[(ysrb['q51'] >= 2), 'her'] = 1

#############################################################################
# Experiences with synth mj, ever

ysrb.loc[(ysrb['q54'] >= 1), 'symj'] = 0 
ysrb.loc[(ysrb['q54'] >= 2), 'symj'] = 1

# use only years 2015 and 2017

ysrb_2yr = ysrb[ysrb['year'] >= 2015][['sex','race4','age','bullied','discord','s_idea','alc30days','mj30d','coc','her']].dropna()

# create dummy variables for sex, race4, and age

ysrb_2yr['Male'] = ysrb_2yr['sex'] - 1

# first is referent
ysrb_2yr['White'] = np.nan
ysrb_2yr['White'] = ysrb_2yr['White'].where(ysrb_2yr['race4']>1,1)
ysrb_2yr['White'] = ysrb_2yr['White'].where(ysrb_2yr['race4']==1,0)
ysrb_2yr['Black'] = np.nan
ysrb_2yr['Black'] = ysrb_2yr['Black'].where(ysrb_2yr['race4']<2,1)
ysrb_2yr['Black'] = ysrb_2yr['Black'].where(ysrb_2yr['race4']>2,1)
ysrb_2yr['Black'] = ysrb_2yr['Black'].where(ysrb_2yr['race4']==2,0)
ysrb_2yr['HispLat'] = np.nan
ysrb_2yr['HispLat'] = ysrb_2yr['HispLat'].where(ysrb_2yr['race4']<3,1)
ysrb_2yr['HispLat'] = ysrb_2yr['HispLat'].where(ysrb_2yr['race4']>3,1)
ysrb_2yr['HispLat'] = ysrb_2yr['HispLat'].where(ysrb_2yr['race4']==3,0)
ysrb_2yr['Other'] = np.nan
ysrb_2yr['Other'] = ysrb_2yr['Other'].where(ysrb_2yr['race4']<4,1)
ysrb_2yr['Other'] = ysrb_2yr['Other'].where(ysrb_2yr['race4']==4,0)

# first is referent
ysrb_2yr['A13u'] = np.nan
ysrb_2yr['A13u'] = ysrb_2yr['A13u'].where(ysrb_2yr['age']>2,1)
ysrb_2yr['A13u'] = ysrb_2yr['A13u'].where(ysrb_2yr['age']<=2,0)
ysrb_2yr['A14'] = np.nan
ysrb_2yr['A14'] = ysrb_2yr['A14'].where(ysrb_2yr['age']<3,1)
ysrb_2yr['A14'] = ysrb_2yr['A14'].where(ysrb_2yr['age']>3,1)
ysrb_2yr['A14'] = ysrb_2yr['A14'].where(ysrb_2yr['age']==3,0)
ysrb_2yr['A15'] = np.nan
ysrb_2yr['A15'] = ysrb_2yr['A15'].where(ysrb_2yr['age']<4,1)
ysrb_2yr['A15'] = ysrb_2yr['A15'].where(ysrb_2yr['age']>4,1)
ysrb_2yr['A15'] = ysrb_2yr['A15'].where(ysrb_2yr['age']==4,0)
ysrb_2yr['A16'] = np.nan
ysrb_2yr['A16'] = ysrb_2yr['A16'].where(ysrb_2yr['age']<5,1)
ysrb_2yr['A16'] = ysrb_2yr['A16'].where(ysrb_2yr['age']>5,1)
ysrb_2yr['A16'] = ysrb_2yr['A16'].where(ysrb_2yr['age']==5,0)
ysrb_2yr['A17'] = np.nan
ysrb_2yr['A17'] = ysrb_2yr['A17'].where(ysrb_2yr['age']<6,1)
ysrb_2yr['A17'] = ysrb_2yr['A17'].where(ysrb_2yr['age']>6,1)
ysrb_2yr['A17'] = ysrb_2yr['A17'].where(ysrb_2yr['age']==6,0)
ysrb_2yr['A18a'] = np.nan
ysrb_2yr['A18a'] = ysrb_2yr['A18a'].where(ysrb_2yr['age']<7,1)
ysrb_2yr['A18a'] = ysrb_2yr['A18a'].where(ysrb_2yr['age']==7,0)

ysrb_2yr = ysrb_2yr.apply(pd.to_numeric)

ysrb_2yr['discuse'] = 0
ysrb_2yr['use'] = 0
ysrb_2yr['disbul'] = 0

ysrb_2yr.loc[(ysrb_2yr['bullied']==1) | (ysrb_2yr['discord']==1) | (ysrb_2yr['alc30days']==1) | (ysrb_2yr['mj30d']==1) | (ysrb_2yr['coc']==1) | (ysrb_2yr['her']==1),'discuse'] = 1
ysrb_2yr.loc[(ysrb_2yr['alc30days']==1) | (ysrb_2yr['mj30d']==1) | (ysrb_2yr['coc']==1) | (ysrb_2yr['her']==1),'use'] = 1
ysrb_2yr['disbul'] = ysrb_2yr['discord']+ysrb_2yr['bullied']-(ysrb_2yr['discord']*ysrb_2yr['bullied'])
X = ysrb_2yr[['Male','Black','HispLat','Other','A14','A15','A16','A17','A18a','bullied','discord','alc30days','mj30d','coc','her']]
y = ysrb_2yr['s_idea']

#os = SMOTE(random_state=0)

#########################################

np.random.seed(19680801)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0)

#########################################
# continue oversampling

#columns = X_train.columns

#os_data_X,os_data_y = os.fit_sample(X_train, y_train)
#os_data_X = pd.DataFrame(data=os_data_X,columns=columns)
#os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])

#X_train=os_data_X[os_data_X.columns]
#y_train=os_data_y['y']

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

all_ages = ['13 yrs or younger','14 yrs','15 yrs', '16 yrs', '17 yrs', '18 yrs or older']

raceEth = ['Black or African American','Hispanic or Latino','Other races', 'White']

sex = ['female','male']

app_name = 'Youth Risk Behaviour Surveillance System (YRBSS), 2015-2017'

app.layout = html.Div([
    # .container class is fixed, .container.scalable is scalable
    html.Div(children=[
        html.Div(children=[
            html.A(
                html.Img(src="https://www.cdc.gov/healthyyouth/data/yrbs/images/yrbss_logo.jpg"),
                href='https://www.cdc.gov/healthyyouth/data/yrbs', target='_blank',
                style={'float': 'left', 'margin-left':'2%'}
            ),
                html.H1(html.A(
                'Youth Risk Behavior Surveillance System (YRBSS), 2015-2017',
                href='https://www.cdc.gov/healthyyouth/data/yrbs', target='_blank',
                style={'text-decoration':'none', 'color': '#1C3B8B', 'font-family':'arial narrow', 'width':'65%', 'margin-left': '2%'}
            )),

        ], className = 'banner'),
    ], className = 'row',style={"width": "100%", "margin-left": "auto", "margin-right": "auto", 'height':'auto','padding':'1%'}),

    html.Div(id='body', className='container scalable', children=[
        html.Div(
            className='row',
            style={'text-decoration':'none', 'color': 'black', 'font-family':'arial narrow', 'width':'68%', 'margin-left': '14%'},
            children=dcc.Markdown(dedent("""  
            Data were collected as part of the Youth Risk Behavior Surveillance System from the years 2015 and 2017. The sample of **17,186** is limited to respondents from 
            **New York**. **Bullying** for each respondent includes experiences online. **Self-harm** was defined as any indication of thoughts of suicide or attempts to 
            harm and includes reports of acts of self-harm or attempts at suicide. 
            **Discordance** was determined by aligning sexual experiences with sexual identity; those that reported same-sex encounters but identified as heterosexual, 
            or those that reported heterosexual experiences but identified as bisexual or homosexual were coded as discordant. Statistically significant differences in levels of self-harm 
            were found among discordant youth that reported experiences of bullying; discordant youth were more likely to report substance use. Additionally, self-harm was 
            more prevalent among females. [Click here](https://github.com/sigmasigmaiota/DATA608F) to view raw data on GitHub.
            """))
        )]),

    html.Div([html.Div([dcc.Dropdown(id='product-selected3',
                                 options=[{'label': i, 'value': i} for i in sex],
                                 placeholder="Select Sex"
                                 )],
                        style={"width": "20%","padding":"2%",'font-family':'arial narrow'}),
          html.Div([dcc.Dropdown(id='product-selected2',
                                 options=[{'label': j, 'value': j} for j in raceEth],
                                 placeholder="Select Race/Ethnicity"
                                 )],
                        style={"width": "20%","padding":"2%",'font-family':'arial narrow'}),
          html.Div([dcc.Dropdown(id='product-selected1',
                                 options=[{'label': k, 'value': k} for k in all_ages],
                                 placeholder="Select Age"
                                 )],
                        style={"width": "20%","padding":"2%",'font-family':'arial narrow'}),
          ], style={"width": "100%", "margin-left": "12%", "margin-right": "auto", "display":"flex"}),
    html.Div([html.Div([
        #html.H3('Column 1'),
        dcc.Graph(id='my-graph')
    ], style={"width": "30%","padding":"1%",'font-family':'arial narrow'}),
    html.Div([
        #html.H3('Column 2'),
        dcc.Graph(id='my-graph2'),
    ], style={"width": "30%","padding":"1%",'font-family':'arial narrow'}),
    html.Div([
        #html.H3('Column 3'),
        dcc.Graph(id='my-graph3'),
    ], style={"width": "30%","padding":"1%",'font-family':'arial narrow'}),
    ], className = "row", style={"display":"flex"}),
    html.Div(id='body2', className='container scalable', children=[
        html.Div([
            #html.H3('Column 1'),
            dcc.Graph(id='my-graph4')
        ], style={'font-family':'arial narrow','margin-left':'5%'}),
        html.Div(
                className='row',
                style={'text-decoration':'none', 'color': 'black', 'font-family':'arial narrow', 'margin-right':'5%', 'padding':'2%'},
                children=dcc.Markdown(dedent("""  
                                             ***  
                                             
                                               
                                             ## Binary Logistic Regression  
                                             
                                             ### Results  
                                             
                                             The model, trained on a test set at 70% of the full sample size (n = 17,186), yielded a weighted average accuracy 
                                             of 79%. The confusion matrix at the left illustrates the precision of the model in the test set. 
                                             Dependent variables were binary variables 
                                             coded from substance use (alcohol, marijuana, cocaine and heroin), bullying experience, discordance, 
                                             race, sex, and age category. Demographic factors such as race, sex and age were included as covariates to control for 
                                             variations among these groups. Bullying, sex, and discordance were all found to be significant predictors of 
                                             indications of self-harm (p < 0.001).
                """))
            )], style={"margin-right": "auto", "display":"flex",'background':'#F5F5F7'}
    )
])
                                  
@app.callback(
    dash.dependencies.Output('my-graph', 'figure'),
    [dash.dependencies.Input('product-selected3', 'value'),
     dash.dependencies.Input('product-selected2', 'value'),
    dash.dependencies.Input('product-selected1','value')])


def update_graph(product_selected1, product_selected2, product_selected3):
    
    sexsel = str(product_selected1)
    
    if sexsel == 'female':
        sexsel2 = 1
    elif sexsel == 'male':
        sexsel2 = 2
    else:
        sexsel2 = 99
        sexsel = 'both sexes'

    rethsel = str(product_selected2)
 
    if rethsel == 'White':
        rethsel2 = 1
    elif rethsel == 'Black or African American':
        rethsel2 = 2
    elif rethsel == 'Hispanic or Latino':
        rethsel2 = 3
    elif rethsel == 'Other races':
        rethsel2 = 4
    else:
        rethsel2 = 99
        rethsel = 'all race/ethnicities'

    agesel = str(product_selected3)
    
    if agesel == '13 yrs or younger':
        agesel2 = 2
    elif agesel == '14 yrs':
        agesel2 = 3
    elif agesel == '15 yrs':
        agesel2 = 4
    elif agesel == '16 yrs':
        agesel2 = 5
    elif agesel == '17 yrs':
        agesel2 = 6
    elif agesel == '18 yrs or older':
        agesel2 = 7        
    else:
        agesel2 = 99
        agesel = 'All ages'
        
    if (sexsel2 == 99) & (rethsel2 != 99) & (agesel2 != 99):
        
        df = ysrb_2yr[(ysrb_2yr['race4']==rethsel2) & (ysrb_2yr['age']==agesel2)]
        
        sprop = 100*round(len(df.index)/len(ysrb_2yr.index),2)
             
        discbull = df['disbul'].sum()/len(df.index)
            
        trace1 = go.Bar(name='bullied',x=[2],y=[df['bullied'].mean()],marker_color='#4e305e')
        trace2 = go.Bar(name='discordant',x=[3],y=[df['discord'].mean()],marker_color='#8a7090')
        trace3 = go.Bar(name='bullied_&_discordant',x=[1],y=[discbull],marker_color='#4f5d75')
        
    elif (sexsel2 != 99) & (rethsel2 == 99) & (agesel2 != 99):
        
        df = ysrb_2yr[(ysrb_2yr['sex']==sexsel2) & (ysrb_2yr['age']==agesel2)]

        sprop = 100*round(len(df.index)/len(ysrb_2yr.index),2)
             
        discbull = df['disbul'].sum()/len(df.index)
            
        trace1 = go.Bar(name='bullied',x=[2],y=[df['bullied'].mean()],marker_color='#4e305e')
        trace2 = go.Bar(name='discordant',x=[3],y=[df['discord'].mean()],marker_color='#8a7090')
        trace3 = go.Bar(name='bullied_&_discordant',x=[1],y=[discbull],marker_color='#4f5d75')
        
    elif (sexsel2 != 99) & (rethsel2 != 99) & (agesel2 == 99):
        
        df = ysrb_2yr[(ysrb_2yr['race4']==rethsel2) & (ysrb_2yr['sex']==sexsel2)]
        
        sprop = 100*round(len(df.index)/len(ysrb_2yr.index),2)
             
        discbull = df['disbul'].sum()/len(df.index)
            
        trace1 = go.Bar(name='bullied',x=[2],y=[df['bullied'].mean()],marker_color='#4e305e')
        trace2 = go.Bar(name='discordant',x=[3],y=[df['discord'].mean()],marker_color='#8a7090')
        trace3 = go.Bar(name='bullied_&_discordant',x=[1],y=[discbull],marker_color='#4f5d75')
        
    elif (sexsel2 != 99) & (rethsel2 == 99) & (agesel2 == 99):
        
        df = ysrb_2yr[ysrb_2yr['sex']==sexsel2]
        
        sprop = 100*round(len(df.index)/len(ysrb_2yr.index),2)
             
        discbull = df['disbul'].sum()/len(df.index)
            
        trace1 = go.Bar(name='bullied',x=[2],y=[df['bullied'].mean()],marker_color='#4e305e')
        trace2 = go.Bar(name='discordant',x=[3],y=[df['discord'].mean()],marker_color='#8a7090')
        trace3 = go.Bar(name='bullied_&_discordant',x=[1],y=[discbull],marker_color='#4f5d75')
        
    elif (sexsel2 == 99) & (rethsel2 != 99) & (agesel2 == 99):
        
        df = ysrb_2yr[ysrb_2yr['race4']==rethsel2]
        
        sprop = 100*round(len(df.index)/len(ysrb_2yr.index),2)
             
        discbull = df['disbul'].sum()/len(df.index)
            
        trace1 = go.Bar(name='bullied',x=[2],y=[df['bullied'].mean()],marker_color='#4e305e')
        trace2 = go.Bar(name='discordant',x=[3],y=[df['discord'].mean()],marker_color='#8a7090')
        trace3 = go.Bar(name='bullied_&_discordant',x=[1],y=[discbull],marker_color='#4f5d75')
        
    elif (sexsel2 == 99) & (rethsel2 == 99) & (agesel2 != 99):
        
        df = ysrb_2yr[ysrb_2yr['age']==agesel2]
        sprop = 100*round(len(df.index)/len(ysrb_2yr.index),2)
             
        discbull = df['disbul'].sum()/len(df.index)
            
        trace1 = go.Bar(name='bullied',x=[2],y=[df['bullied'].mean()],marker_color='#4e305e')
        trace2 = go.Bar(name='discordant',x=[3],y=[df['discord'].mean()],marker_color='#8a7090')
        trace3 = go.Bar(name='bullied_&_discordant',x=[1],y=[discbull],marker_color='#4f5d75')
        
    elif (sexsel2 != 99) & (rethsel2 != 99) & (agesel2 != 99):
        
        df = ysrb_2yr[(ysrb_2yr['sex']==sexsel2) & (ysrb_2yr['race4']==rethsel2) & (ysrb_2yr['age']==agesel2)]
        sprop = 100*round(len(df.index)/len(ysrb_2yr.index),2)
             
        discbull = df['disbul'].sum()/len(df.index)
            
        trace1 = go.Bar(name='bullied',x=[2],y=[df['bullied'].mean()],marker_color='#4e305e')
        trace2 = go.Bar(name='discordant',x=[3],y=[df['discord'].mean()],marker_color='#8a7090')
        trace3 = go.Bar(name='bullied_&_discordant',x=[1],y=[discbull],marker_color='#4f5d75')
        
    else:
        
        df = ysrb_2yr
        
        #'#8a7090','#89a7a7','#72e1d1','#b5d8cc'
        
        sprop = 100*round(len(df.index)/len(ysrb_2yr.index),2)
             
        discbull = df['disbul'].sum()/len(df.index)
            
        trace1 = go.Bar(name='bullied',x=[2],y=[df['bullied'].mean()],marker_color='#4e305e')
        trace2 = go.Bar(name='discordant',x=[3],y=[df['discord'].mean()],marker_color='#8a7090')
        trace3 = go.Bar(name='bullied_&_discordant',x=[1],y=[discbull],marker_color='#4f5d75')

    return {
        'data': [trace1, trace2, trace3],
        'layout': go.Layout(
  
            title=f'{agesel}, {rethsel}, {sexsel}'+'<br>'+f'{sprop}'+'% of total sample',
            showlegend = False,
            yaxis_title_text='proportion',
            yaxis_tickformat = '%',
            yaxis = dict(
                range = [0,0.35]
                ),
            xaxis = dict(
                tickmode = 'array',
                tickvals = [1, 2, 3],
                ticktext = ['bullied or discordant', 'bullied', 'discordant', ]
            ),
            font=dict(
                family="Arial Narrow, sans-serif"
            )

        )
    }

@app.callback(
    dash.dependencies.Output('my-graph2', 'figure'),
    [dash.dependencies.Input('product-selected3', 'value'),
     dash.dependencies.Input('product-selected2', 'value'),
    dash.dependencies.Input('product-selected1','value')])


def update_graph2(product_selected1, product_selected2, product_selected3):
    
    sexsel_1 = str(product_selected1)
    
    if sexsel_1 == 'female':
        sexsel2_1 = 1
    elif sexsel_1 == 'male':
        sexsel2_1 = 2
    else:
        sexsel2_1 = 99
        sexsel_1 = 'both sexes'

    rethsel_1 = str(product_selected2)
    
    if rethsel_1 == 'White':
        rethsel2_1 = 1
    elif rethsel_1 == 'Black or African American':
        rethsel2_1 = 2
    elif rethsel_1 == 'Hispanic or Latino':
        rethsel2_1 = 3
    elif rethsel_1 == 'Other races':
        rethsel2_1 = 4
    else:
        rethsel2_1 = 99
        rethsel_1 = 'all race/ethnicities'

    agesel_1 = str(product_selected3)
    
    if agesel_1 == '13 yrs or younger':
        agesel2_1 = 2
    elif agesel_1 == '14 yrs':
        agesel2_1 = 3
    elif agesel_1 == '15 yrs':
        agesel2_1 = 4
    elif agesel_1 == '16 yrs':
        agesel2_1 = 5
    elif agesel_1 == '17 yrs':
        agesel2_1 = 6
    elif agesel_1 == '18 yrs or older':
        agesel2_1 = 7        
    else:
        agesel2_1 = 99
        agesel_1 = 'All ages'
        
    if (sexsel2_1 == 99) & (rethsel2_1 != 99) & (agesel2_1 != 99):
        
        df_1 = ysrb_2yr[(ysrb_2yr['race4']==rethsel2_1) & (ysrb_2yr['age']==agesel2_1)]
        
        sprop_1 = 100*round(len(df_1.index)/len(ysrb_2yr.index),2)
    
        trace4 = go.Bar(name='alcohol', x=[2], y=[df_1['alc30days'].mean()], marker_color='#ecce8e')
        trace5 = go.Bar(name='marijuana', x=[3], y=[df_1['mj30d'].mean()], marker_color='#dbcf96')
        trace6 = go.Bar(name='cocaine', x=[4], y=[df_1['coc'].mean()], marker_color='#c2c6a7')
        trace7 = go.Bar(name='heroin', x=[5], y=[df_1['her'].mean()], marker_color='#bfc0c0')
        trace8 = go.Bar(name='any', x=[1], y=[df_1['use'].mean()], marker_color='#a98743')
        
    elif (sexsel2_1 != 99) & (rethsel2_1 == 99) & (agesel2_1 != 99):
        
        df_1 = ysrb_2yr[(ysrb_2yr['sex']==sexsel2_1) & (ysrb_2yr['age']==agesel2_1)]
        
        sprop_1 = 100*round(len(df_1.index)/len(ysrb_2yr.index),2)
    
        trace4 = go.Bar(name='alcohol', x=[2], y=[df_1['alc30days'].mean()], marker_color='#ecce8e')
        trace5 = go.Bar(name='marijuana', x=[3], y=[df_1['mj30d'].mean()], marker_color='#dbcf96')
        trace6 = go.Bar(name='cocaine', x=[4], y=[df_1['coc'].mean()], marker_color='#c2c6a7')
        trace7 = go.Bar(name='heroin', x=[5], y=[df_1['her'].mean()], marker_color='#bfc0c0')
        trace8 = go.Bar(name='any', x=[1], y=[df_1['use'].mean()], marker_color='#a98743')
        
    elif (sexsel2_1 != 99) & (rethsel2_1 != 99) & (agesel2_1 == 99):
        
        df_1 = ysrb_2yr[(ysrb_2yr['race4']==rethsel2_1) & (ysrb_2yr['sex']==sexsel2_1)]
        
        sprop_1 = 100*round(len(df_1.index)/len(ysrb_2yr.index),2)
    
        trace4 = go.Bar(name='alcohol', x=[2], y=[df_1['alc30days'].mean()], marker_color='#ecce8e')
        trace5 = go.Bar(name='marijuana', x=[3], y=[df_1['mj30d'].mean()], marker_color='#dbcf96')
        trace6 = go.Bar(name='cocaine', x=[4], y=[df_1['coc'].mean()], marker_color='#c2c6a7')
        trace7 = go.Bar(name='heroin', x=[5], y=[df_1['her'].mean()], marker_color='#bfc0c0')
        trace8 = go.Bar(name='any', x=[1], y=[df_1['use'].mean()], marker_color='#a98743')
        
    elif (sexsel2_1 != 99) & (rethsel2_1 == 99) & (agesel2_1 == 99):
        
        df_1 = ysrb_2yr[ysrb_2yr['sex']==sexsel2_1]
        
        sprop_1 = 100*round(len(df_1.index)/len(ysrb_2yr.index),2)
    
        trace4 = go.Bar(name='alcohol', x=[2], y=[df_1['alc30days'].mean()], marker_color='#ecce8e')
        trace5 = go.Bar(name='marijuana', x=[3], y=[df_1['mj30d'].mean()], marker_color='#dbcf96')
        trace6 = go.Bar(name='cocaine', x=[4], y=[df_1['coc'].mean()], marker_color='#c2c6a7')
        trace7 = go.Bar(name='heroin', x=[5], y=[df_1['her'].mean()], marker_color='#bfc0c0')
        trace8 = go.Bar(name='any', x=[1], y=[df_1['use'].mean()], marker_color='#a98743')
        
    elif (sexsel2_1 == 99) & (rethsel2_1 != 99) & (agesel2_1 == 99):
        
        df_1 = ysrb_2yr[ysrb_2yr['race4']==rethsel2_1]
        
        sprop_1 = 100*round(len(df_1.index)/len(ysrb_2yr.index),2)
    
        trace4 = go.Bar(name='alcohol', x=[2], y=[df_1['alc30days'].mean()], marker_color='#ecce8e')
        trace5 = go.Bar(name='marijuana', x=[3], y=[df_1['mj30d'].mean()], marker_color='#dbcf96')
        trace6 = go.Bar(name='cocaine', x=[4], y=[df_1['coc'].mean()], marker_color='#c2c6a7')
        trace7 = go.Bar(name='heroin', x=[5], y=[df_1['her'].mean()], marker_color='#bfc0c0')
        trace8 = go.Bar(name='any', x=[1], y=[df_1['use'].mean()], marker_color='#a98743')
        
    elif (sexsel2_1 == 99) & (rethsel2_1 == 99) & (agesel2_1 != 99):
        
        df_1 = ysrb_2yr[ysrb_2yr['age']==agesel2_1]
        
        sprop_1 = 100*round(len(df_1.index)/len(ysrb_2yr.index),2)
    
        trace4 = go.Bar(name='alcohol', x=[2], y=[df_1['alc30days'].mean()], marker_color='#ecce8e')
        trace5 = go.Bar(name='marijuana', x=[3], y=[df_1['mj30d'].mean()], marker_color='#dbcf96')
        trace6 = go.Bar(name='cocaine', x=[4], y=[df_1['coc'].mean()], marker_color='#c2c6a7')
        trace7 = go.Bar(name='heroin', x=[5], y=[df_1['her'].mean()], marker_color='#bfc0c0')
        trace8 = go.Bar(name='any', x=[1], y=[df_1['use'].mean()], marker_color='#a98743')
        
    elif (sexsel2_1 != 99) & (rethsel2_1 != 99) & (agesel2_1 != 99):
        
        df_1 = ysrb_2yr[(ysrb_2yr['sex']==sexsel2_1) & (ysrb_2yr['race4']==rethsel2_1) & (ysrb_2yr['age']==agesel2_1)]
        
        sprop_1 = 100*round(len(df_1.index)/len(ysrb_2yr.index),2)
    
        trace4 = go.Bar(name='alcohol', x=[2], y=[df_1['alc30days'].mean()], marker_color='#ecce8e')
        trace5 = go.Bar(name='marijuana', x=[3], y=[df_1['mj30d'].mean()], marker_color='#dbcf96')
        trace6 = go.Bar(name='cocaine', x=[4], y=[df_1['coc'].mean()], marker_color='#c2c6a7')
        trace7 = go.Bar(name='heroin', x=[5], y=[df_1['her'].mean()], marker_color='#bfc0c0')
        trace8 = go.Bar(name='any', x=[1], y=[df_1['use'].mean()], marker_color='#a98743')
        
    else:
        
        df_1 = ysrb_2yr
        
        #colors = ['#ecce8e','#dbcf96','#c2c6a7','#9ac2c5']
        
        sprop_1 = 100*round(len(df_1.index)/len(ysrb_2yr.index),2)
    
        trace4 = go.Bar(name='alcohol', x=[2], y=[df_1['alc30days'].mean()], marker_color='#ecce8e')
        trace5 = go.Bar(name='marijuana', x=[3], y=[df_1['mj30d'].mean()], marker_color='#dbcf96')
        trace6 = go.Bar(name='cocaine', x=[4], y=[df_1['coc'].mean()], marker_color='#c2c6a7')
        trace7 = go.Bar(name='heroin', x=[5], y=[df_1['her'].mean()], marker_color='#bfc0c0')
        trace8 = go.Bar(name='any', x=[1], y=[df_1['use'].mean()], marker_color='#a98743')

    return {
        'data': [trace4, trace5, trace6, trace7, trace8],
        'layout': go.Layout(
  
            title=f'{agesel_1}, {rethsel_1}, {sexsel_1}'+'<br>'+f'{sprop_1}'+'% of total sample',
            showlegend = False,
            yaxis = dict(
                range = [0,0.35]
                ),
            yaxis_title_text='proportion',
            yaxis_tickformat = '%',
            xaxis = dict(
                tickmode = 'array',
                tickvals = [1, 2, 3, 4, 5],
                ticktext = ['any use', 'alcohol', 'marijuana', 'cocaine', 'heroin']
            ),
            font=dict(
                family="Arial Narrow, sans-serif"
            )

        )
    }

@app.callback(
    dash.dependencies.Output('my-graph3', 'figure'),
    [dash.dependencies.Input('product-selected3', 'value'),
     dash.dependencies.Input('product-selected2', 'value'),
    dash.dependencies.Input('product-selected1','value')])


def update_graph3(product_selected1, product_selected2, product_selected3):
    
    sexsel_2 = str(product_selected1)
    
    if sexsel_2 == 'female':
        sexsel2_2 = 1
    elif sexsel_2 == 'male':
        sexsel2_2 = 2
    else:
        sexsel2_2 = 99
        sexsel_2 = 'both sexes'

    rethsel_2 = str(product_selected2)
    
    if rethsel_2 == 'White':
        rethsel2_2 = 1
    elif rethsel_2 == 'Black or African American':
        rethsel2_2 = 2
    elif rethsel_2 == 'Hispanic or Latino':
        rethsel2_2 = 3
    elif rethsel_2 == 'Other races':
        rethsel2_2 = 4
    else:
        rethsel2_2 = 99
        rethsel_2 = 'all race/ethnicities'

    agesel_2 = str(product_selected3)
    
    if agesel_2 == '13 yrs or younger':
        agesel2_2 = 2
    elif agesel_2 == '14 yrs':
        agesel2_2 = 3
    elif agesel_2 == '15 yrs':
        agesel2_2 = 4
    elif agesel_2 == '16 yrs':
        agesel2_2 = 5
    elif agesel_2 == '17 yrs':
        agesel2_2 = 6
    elif agesel_2 == '18 yrs or older':
        agesel2_2 = 7        
    else:
        agesel2_2 = 99
        agesel_2 = 'All ages'
        
    if (sexsel2_2 == 99) & (rethsel2_2 != 99) & (agesel2_2 != 99):
        
        df_2 = ysrb_2yr[(ysrb_2yr['race4']==rethsel2_2) & (ysrb_2yr['age']==agesel2_2)]
        
        labels4 = ['Self-harm','No self-harm']
        
        sprop_2 = 100*round(df_2.discuse.sum()/len(df_2.index),2)
        
        shtot = df_2[df_2['discuse']==1]['s_idea'].sum()
        
        tot = len(df_2.index)
        
        values4 = [shtot, tot-shtot]
            
        trace11 = go.Pie(labels=labels4, values=values4, pull=[0,0.2], marker=dict(colors=['#95B8D1','#B8E0D2']))
        
    elif (sexsel2_2 != 99) & (rethsel2_2 == 99) & (agesel2_2 != 99):
        
        df_2 = ysrb_2yr[(ysrb_2yr['sex']==sexsel2_2) & (ysrb_2yr['age']==agesel2_2)]
        
        labels4 = ['Self-harm','No self-harm']
        
        sprop_2 = 100*round(df_2.discuse.sum()/len(df_2.index),2)
        
        shtot = df_2[df_2['discuse']==1]['s_idea'].sum()
        
        tot = len(df_2.index)
        
        values4 = [shtot, tot-shtot]
            
        trace11 = go.Pie(labels=labels4, values=values4, pull=[0,0.2], marker=dict(colors=['#95B8D1','#B8E0D2']))
        
    elif (sexsel2_2 != 99) & (rethsel2_2 != 99) & (agesel2_2 == 99):
        
        df_2 = ysrb_2yr[(ysrb_2yr['race4']==rethsel2_2) & (ysrb_2yr['sex']==sexsel2_2)]
        
        labels4 = ['Self-harm','No self-harm']
        
        sprop_2 = 100*round(df_2.discuse.sum()/len(df_2.index),2)
        
        shtot = df_2[df_2['discuse']==1]['s_idea'].sum()
        
        tot = len(df_2.index)
        
        values4 = [shtot, tot-shtot]
            
        trace11 = go.Pie(labels=labels4, values=values4, pull=[0,0.2], marker=dict(colors=['#95B8D1','#B8E0D2']))
        
    elif (sexsel2_2 != 99) & (rethsel2_2 == 99) & (agesel2_2 == 99):
        
        df_2 = ysrb_2yr[ysrb_2yr['sex']==sexsel2_2]
        
        labels4 = ['Self-harm','No self-harm']
        
        sprop_2 = 100*round(df_2.discuse.sum()/len(df_2.index),2)
        
        shtot = df_2[df_2['discuse']==1]['s_idea'].sum()
        
        tot = len(df_2.index)
        
        values4 = [shtot, tot-shtot]
            
        trace11 = go.Pie(labels=labels4, values=values4, pull=[0,0.2], marker=dict(colors=['#95B8D1','#B8E0D2']))
        
    elif (sexsel2_2 == 99) & (rethsel2_2 != 99) & (agesel2_2 == 99):
        
        df_2 = ysrb_2yr[ysrb_2yr['race4']==rethsel2_2]
        
        labels4 = ['Self-harm','No self-harm']
        
        sprop_2 = 100*round(df_2.discuse.sum()/len(df_2.index),2)
        
        shtot = df_2[df_2['discuse']==1]['s_idea'].sum()
        
        tot = len(df_2.index)
        
        values4 = [shtot, tot-shtot]
            
        trace11 = go.Pie(labels=labels4, values=values4, pull=[0,0.2], marker=dict(colors=['#95B8D1','#B8E0D2']))
        
    elif (sexsel2_2 == 99) & (rethsel2_2 == 99) & (agesel2_2 != 99):
        
        df_2 = ysrb_2yr[ysrb_2yr['age']==agesel2_2]
        
        labels4 = ['Self-harm','No self-harm']
        
        sprop_2 = 100*round(df_2.discuse.sum()/len(df_2.index),2)
        
        shtot = df_2[df_2['discuse']==1]['s_idea'].sum()
        
        tot = len(df_2.index)
        
        values4 = [shtot, tot-shtot]
            
        trace11 = go.Pie(labels=labels4, values=values4, pull=[0,0.2], marker=dict(colors=['#95B8D1','#B8E0D2']))
        
    elif (sexsel2_2 != 99) & (rethsel2_2 != 99) & (agesel2_2 != 99):
        
        df_2 = ysrb_2yr[(ysrb_2yr['sex']==sexsel2_2) & (ysrb_2yr['race4']==rethsel2_2) & (ysrb_2yr['age']==agesel2_2)]
        
        labels4 = ['Self-harm','No self-harm']
        
        sprop_2 = 100*round(df_2.discuse.sum()/len(df_2.index),2)
        
        shtot = df_2[df_2['discuse']==1]['s_idea'].sum()
        
        tot = len(df_2.index)
        
        values4 = [shtot, tot-shtot]
            
        trace11 = go.Pie(labels=labels4, values=values4, pull=[0,0.2], marker=dict(colors=['#95B8D1','#B8E0D2']))
        
    else:
        
        df_2 = ysrb_2yr
        
        #colors = ['#ecce8e','#dbcf96','#c2c6a7','#9ac2c5']

        labels4 = ['Self-harm','No self-harm']
        
        sprop_2 = 100*round(df_2.discuse.sum()/len(df_2.index),2)
        
        shtot = df_2[df_2['discuse']==1]['s_idea'].sum()
        
        tot = len(df_2.index)
        
        values4 = [shtot, tot-shtot]
            
        trace11 = go.Pie(labels=labels4, values=values4, pull=[0,0.2], marker=dict(colors=['#95B8D1','#B8E0D2']))

    return {
        'data': [trace11],
        'layout': go.Layout(
  
            title='Self-harm, of those '+f'{sprop_2}'+'% who were'+'<br>Bullied, Discordant, or Substance Users',
            font=dict(
                family="Arial Narrow, sans-serif"
                )

        )
    }

@app.callback(
    dash.dependencies.Output('my-graph4', 'figure'),
    [dash.dependencies.Input('product-selected3', 'value'),
     dash.dependencies.Input('product-selected2', 'value'),
    dash.dependencies.Input('product-selected1','value')])

def update_graph4(product_selected1, product_selected2, product_selected3):
    
    z = confusion_matrix(y_test, y_pred)
    
    trace15 = go.Heatmap(z=z, x = [ '0', '1'], y = [ '1', '0'], colorscale = ["#ab6d41", "#ffb785"])
    
    xx = [ 0, 1]
    yy = [1, 0]
    
    annotations = go.Annotations()
    for n, row in enumerate(z):
        for m, val in enumerate(row):
            annotations.append(go.Annotation(text=str(z[n][m]), x=xx[m], y=yy[n],
                                             xref='x1', yref='y1', showarrow=False))



    return {
        'data': [trace15],
        'layout': go.Layout(
  
            font=dict(
                family="Arial Narrow, sans-serif",
                color="black",
                size=16
                ),
            annotations=annotations,
            width=400,
            height=400,
            autosize=False,
            showlegend=False,
            yaxis = dict(
                tickmode = 'array',
                tickvals = [-0.5,0,0.5,1,1.5],
                ticktext = ['', 'Self-harm', '', 'No<br>self-harm', '']
                ),
            xaxis = dict(
                tickmode = 'array',
                tickvals = [-0.5,0,0.5,1,1.5],
                ticktext = ['', 'No self-harm', '', 'Self-harm', '']
            ),
            title = '<b>Confusion Matrix</b>',
            yaxis_title_text='<b>actual</b>',
            xaxis_title_text='<b>predicted</b>',
            paper_bgcolor='#F5F5F7'
        )
    }



# Running the server
if __name__ == '__main__':
    app.run_server(debug=True)
    #port = int(os.environ.get('PORT', 5000))
    #app.run_server(host='0.0.0.0',port=port)
