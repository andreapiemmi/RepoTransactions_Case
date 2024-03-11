#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 14:55:48 2024

@author: andreapiemontese
"""
from dash import Dash, html, dcc, Input, Output, dash_table
import plotly.express as px
import dash_daq as daq
import dash_bootstrap_components as dbc          # pip install dash-bootstrap-components
import pandas as pd
import numpy as np
from datetime import date,datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import statsmodels.formula.api as smf



calendar={1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
          7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

df=pd.read_csv('https://raw.githubusercontent.com/andreapiemmi/RepoTransactions_Case/main/repo_trades.csv',sep=';')
df[df.columns[['Date' in c for c in df.columns]]]=df.loc[:,['Date' in c for c in df.columns]].apply(
    lambda x: pd.to_datetime(x.str.replace('.','-',regex=True)),axis=0)
df=df.sort_values(by=['BusinessDate','PurchaseDate']).reset_index(drop=True)

def q3(s):
    return s.quantile(0.75)
def q1(s):
  return s.quantile(0.25)
def standardize(x):
  return (x-x.mean())/x.std()

def Question1():
    d1=pd.DataFrame()
    for i in pd.date_range(date(2020,1,1), periods=366):
        t=df.loc[np.logical_and(df['PurchaseDate']<=i,
                                  i<df['RepurchaseDate'])][['Term','CashAmount']].groupby(['Term']).sum().squeeze().rename(i)
        d1=pd.concat([d1,pd.DataFrame(t).T],axis=0)

    fig = go.Figure()
    for i in d1.columns:
        fig.add_trace(go.Scatter(
            x=d1.index, y=d1[i],
            mode='lines',
            line=dict(width=0.5),
            name=i,
            stackgroup='one',
            text=d1.columns,
            customdata=d1.columns
        ))
    fig.add_trace(go.Scatter(x=d1.index,y=d1.sum(axis=1),name='Total',line=dict(color='slategray',dash='dot')))
    fig.update_traces(hovertemplate='<b>%{y}</b>')
    fig.update_layout(xaxis_title='Date',
                      yaxis_title='Value in CHF',
                      hovermode="x",
                      template='plotly_dark', plot_bgcolor= 'rgba(0, 0, 0, 0)',paper_bgcolor= 'rgba(0, 0, 0, 0)')
    return fig,d1

def Question2():
    selection_Q2=['ON','1W','1M']
    d2=pd.DataFrame()
    for i in pd.date_range(date(2020,1,1), periods=366):
        t=df.loc[np.logical_and(df['PurchaseDate']<=i,
                                  i<df['RepurchaseDate'])][['Term','CashAmount','Rate']].groupby(['Term']).apply(
                                      lambda x: np.dot(x['CashAmount']/x['CashAmount'].sum(), x['Rate']).sum() ).squeeze().rename(i)

        #print(t)
        d2=pd.concat([d2,pd.DataFrame(t).T],axis=0)
    d2=d2[selection_Q2]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=d2.index,y=d2['ON'],name='ON',line=dict(color='teal')))
    fig.add_trace(go.Scatter(x=d2.index,y=d2['1W'],name='1W',line=dict(color='darkturquoise')))
    fig.add_trace(go.Scatter(x=d2.index,y=d2['1M'],name='1M',line=dict(color='darkseagreen')))
    fig.update_traces(hovertemplate='<b>%{y}</b>')
    fig.update_layout(xaxis_title='Date',
                      yaxis_title='Rate <b>p.a. (%)</b>',
                      hovermode="x",
                      template='plotly_dark', plot_bgcolor= 'rgba(0, 0, 0, 0)',paper_bgcolor= 'rgba(0, 0, 0, 0)')

    return fig,d2


def Question3():
    d2=Question2()[1]
    d3=pd.DataFrame()

    functions={'Rate':['std', q1, q3]}

    for i in pd.date_range(date(2020,1,1), periods=366):
        t=df[df['Term']=='ON'].loc[np.logical_and(df['PurchaseDate']<=i,
                                  i<df['RepurchaseDate'])][['Term','Rate']].groupby(['Term']).agg(functions).T.reset_index(
                                      level=[0]).drop(columns='level_0').rename(columns={'ON':i}).T
        d3=pd.concat([d3,pd.DataFrame(t)],axis=0)
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])

    fig1_p2=go.Figure(go.Scatter(x=d2['ON'].index,y=d2['ON'].values,mode='lines',name='idIR',line_color='slategray',yaxis="y2",opacity=0.4))
    fig1_p2.add_traces(go.Scatter(x=d3['q3'].index,y=d3['q3'].values,fill=None,line=dict(width=0),name='75%',
                                  yaxis="y2",line_color='slategray',opacity=0.15))
    fig1_p2.add_traces(go.Scatter(x=d3['q1'].index,y=d3['q1'].values,fill='tonexty',fillcolor='rgba(112, 128, 144, 0.1)',
                                  line=dict(width=0),name='25%',yaxis="y2",line_color='slategray',opacity=0.15))

    fig1_p1=px.line(d3['std'], render_mode="webgl",).update_traces(line_color='teal')

    fig1.add_traces(fig1_p1.data + fig1_p2.data).update_layout(showlegend=True,
                           template='plotly_white')
    fig1.update_traces(hovertemplate='%{y:.2f}')
    fig1.layout.yaxis.title="Volatility (Intraday)<b>(%)</b>"
    fig1.layout.yaxis.color="teal"
    fig1.layout.yaxis2.title="Mean Interest Rate (Intraday) <b>p.a.(%)</b>"
    fig1.layout.yaxis2.color = 'darkgray'
    fig1.update_layout(hovermode="x unified",template='plotly_dark',
                      hoverlabel=dict(bgcolor='rgba(255,255,255,0.1)',font=dict(color='white')),
                      plot_bgcolor= 'rgba(0, 0, 0, 0)',paper_bgcolor= 'rgba(0, 0, 0, 0)')
    return fig1

def Question4_d4():
    d1, d2= Question1()[1], Question2()[1]
    d4=pd.DataFrame()
    for i in pd.date_range(date(2020,1,1), periods=366):
      t1=df.loc[np.logical_and(df['PurchaseDate']<=i,
                                  i<df['RepurchaseDate'])][['Term','CashAmount']].groupby(['Term']).skew().squeeze().rename(i)
      t2=df.loc[np.logical_and(df['PurchaseDate']<=i,
                                  i<df['RepurchaseDate'])][['Term','Rate']].groupby(['Term']).skew().squeeze().rename(i)
      if 'ON' in t1.index or 'ON' in t2.index:
        temp=pd.concat([t1,t2],axis=1).T['ON'].rename(i).reset_index(drop=True).set_axis(['ON_Volume_skew','ON_Rate_skew'])
        d4=pd.concat([d4,pd.DataFrame(temp).T],axis=0)

    d4=d4.join(pd.concat([d1['ON'].rename('ON_Total_Volume'),d2['ON'].rename('ON_Average_Rate')],axis=1),how='left')
    for i in d4.index:
      d4.loc[i,'ON_N_repos']=df[df['Term']=='ON'].loc[np.logical_and(df['PurchaseDate']<=i,
                                  i<df['RepurchaseDate'])][['Term','CashAmount']].shape[0]


    Month_Ranges=list(zip( [np.min(d4.index[d4.index.month==m]) for m in np.unique(d4.index.month)],
     [np.max(d4.index[d4.index.month==m]) for m in np.unique(d4.index.month)] ))
    d4['Month_End_Dummy0']=[*pd.Series([*d4.index]).apply(lambda x: 1 if x in [item[1] for item in Month_Ranges] else 0)]

    for lag in range(1,7):
      Range_for_Ones=list( zip(d4['Month_End_Dummy0'][d4['Month_End_Dummy0']==1].index-timedelta(days=lag),d4.index[d4['Month_End_Dummy0']==1]) )
      for i in range(len(Range_for_Ones)):
        selection=d4.index[np.logical_and(d4.index>=Range_for_Ones[i][0],
                                         d4.index<=Range_for_Ones[i][1])]
        d4.loc[selection,'Month_End_Dummy'+str(lag)]=1


    d4[d4.columns[['Dummy' in s for s in d4.columns]]]=d4[d4.columns[['Dummy' in s for s in d4.columns]]].fillna(0)

    functions={'CashAmount':['median', q1, q3]}
    dd=pd.DataFrame()
    for i in pd.date_range(date(2020,1,1), periods=366):
        t=df[df['Term']=='ON'].loc[np.logical_and(df['PurchaseDate']<=i,
                                  i<df['RepurchaseDate'])][['Term','CashAmount']].groupby(['Term']).agg(functions).T.reset_index(
                                      level=[0]).drop(columns='level_0').rename(columns={'ON':i}).T
        dd=pd.concat([dd,t],axis=0)
    dd=pd.concat([d4,dd],axis=1)

    fig=go.Figure()
    for month in np.arange(1,13):
      if month!=12:
        switch='legendonly'
      else:
        switch=True
      t=dd[dd.index.month==month].reset_index().set_index(np.arange(1,len(dd[dd.index.month==month])+1)/dd[dd.index.month==month].shape[0])
      fig.add_trace(go.Scatter(x=t['ON_Total_Volume'].index,y=t['ON_Total_Volume']/t['ON_N_repos'],name=str(calendar[month]),
                               visible=switch))
      fig.add_traces(go.Scatter(x=t['q3'].index,y=t['q3'].values,fill=None,line=dict(width=0),name=str(calendar[month])+'75%',
                                  line_color='slategray',opacity=0.15,visible=switch))
      fig.add_traces(go.Scatter(x=t['q1'].index,y=t['q1'].values,fill='tonexty',fillcolor='rgba(112, 128, 144, 0.1)',
                                  line=dict(width=0),name=str(calendar[month])+'25%',line_color='slategray',opacity=0.15,visible=switch))
      fig.add_traces(go.Scatter(x=t['median'].index,y=t['median'].values,name=str(calendar[month])+'50%',visible=switch))
      fig.update_layout(hovermode="x",template='plotly_dark',
                      yaxis_title='Oustanding Amount',hoverlabel=dict(bgcolor='rgba(255,255,255,0.1)'),
                      plot_bgcolor= 'rgba(0, 0, 0, 0)',paper_bgcolor= 'rgba(0, 0, 0, 0)')

    return d4, fig

def Question5_d5():
    d5=pd.DataFrame()
    for i in pd.date_range(date(2020,1,1), periods=366):
        t1=df.loc[np.logical_and(df['PurchaseDate']<=i,
                                  i<df['RepurchaseDate'])][['CashTakerID','Term','CashAmount']].groupby(['CashTakerID','Term']).sum()

        t2=df.loc[np.logical_and(df['PurchaseDate']<=i,
                                  i<df['RepurchaseDate'])][['CashTakerID','Term','CashAmount','Rate']].groupby(['CashTakerID','Term']).apply(
                                      lambda x: np.dot(x['CashAmount']/x['CashAmount'].sum(), x['Rate']).sum() ).rename('Rate')

        t=pd.concat([t1,t2],axis=1)
        t['Date']=i
        t=t.loc[[item[1]=='ON' for item in t.index],:].reset_index()
        d5=pd.concat([d5,t],axis=0)
    d5=d5.drop(columns=['Term']).reset_index(drop=True)
    #d5=d5.set_index(['CashTakerID','Date'])
    d5=pd.merge(d5,
             d5[['Date','CashAmount','Rate']].groupby(['Date']).mean().rename(columns={'CashAmount':'CashAmount_Mean-per-Date',
                                                                                       'Rate':'Rate_Mean-per-Date'}) ,
             how='left', left_index=False, right_index=True, left_on='Date')
    return d5


d4=Question4_d4()[0] #**Defined Globally**, otherwise every reaction would take ages
d5=Question5_d5()

app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.layout = dbc.Container([
    html.Br(),
    html.H1("Repo Transactions - Data Explorer", className='mb-2', style={'textAlign':'left'}),
    html.Hr(style = {'size' : '50', 'borderColor':'stonegray','borderHeight': "10vh", "width": "95%",}),
    dbc.Card(
        dbc.CardBody([
            dbc.Row([dbc.Col( [ html.H2("I. Oustanding Volumes by Day (2020)", className='mb-2', style={'textAlign':'left'})
             ])]),
            dbc.Row([
                dcc.Graph(figure=Question1()[0])

            ],justify='left')


        ])
    ),
        dbc.Card(
        dbc.CardBody([
            html.H2("II. Volume Weighted Repo Rates by Day (2020), % p.a. (ON,1W,1M)", className='mb-2', style={'textAlign':'left'}),
            dbc.Row([
                dcc.Graph(figure=Question2()[0])
    ], className='mt-4')
        ]),
    ),

        dbc.Card(
        dbc.CardBody([
            html.H2("III. Intra-Day Dispersion by Day (2020, ON)", className='mb-2', style={'textAlign':'left'}),
            dbc.Row([
                dcc.Graph(figure=Question3())
    ], className='mt-4')
        ]),
    ),

        dbc.Card(
        dbc.CardBody([
            html.H2("IV. Interest Rate & Trading Volumes at Reporting Dates (2020, ON)",
                    className='mb-2', style={'textAlign':'left'}),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='Scatter_MonthEnd')
                    ],width=5),
                dbc.Col([
                    dcc.Graph(figure=Question4_d4()[1])
                    ],width=7)

    ], className='mt-4'),
            dbc.Row([
                dbc.Col([
                    html.Div([html.P('Consideration Period (Days in Adv.)')],style={'font-weight':'bold','font-size':'16px'}),
                    dcc.Slider(0, 6,value=0,step=1,id='my-slider')
                    ]),
                dbc.Col([
                 html.Div([html.P('Exclude December')],
                 style={'width':'30%','display':'inline-block',
                        'margin': '0px 0px 0px 0px','font-weight':'bold',
                        'font-size':'16px'}),
                    html.Div([daq.ToggleSwitch(id='my-toggle-switch',value=True)],style={'width':'40%','display':'inline-block'}),
                    html.Div([html.P('Include December')],
                    style={'width':'30%','display':'inline-block',
                        'margin': '0px 0px 0px 0px','font-weight':'bold',
                        'font-size':'16px'})
                    ])

    ], className='mt-4'),

            dbc.Row([
                dbc.Col([html.P('Linear Regression (OLS)',style={'font-size': '25px'}),
                         dash_table.DataTable(id = 'OLS_Table',
                                              style_data={'color':'aliceblue','backgroundColor':'slategray'},fill_width=False),
                         dcc.RadioItems(id='Tot-v-Avg',options=['Total Outstanding Amount',"Avg Repo's Outstanding Amount"],value='Total Outstanding Amount')
                    ]),
                dbc.Col([html.P('Quantile Regression',style={'font-size': '25px'}),
                         dash_table.DataTable(id = 'QuanReg_Table',
                                              style_data={'color':'aliceblue','backgroundColor':'slategray'},fill_width=False)
                    ])
                ],className='mt-4')
        ]),
    ),

        dbc.Card(
        dbc.CardBody([
            html.H2("V. Differential Interest Rate v. Competitors (2020, ON)", className='mb-2', style={'textAlign':'left'}),
            dbc.Row([

                    html.Div([html.P('CashTakerID')],style={'font-weight':'bold'}),
                    dcc.RadioItems(id='Cash-Taker',options=[*np.arange(1,11)],value=10),





    ], className='mt-4'),
    dbc.Row([
        dcc.Graph(id='Fixed_Effects')], className='mt-4')
        ]),
    )


])



@app.callback(
    Output('Scatter_MonthEnd', 'figure'),
    Input('my-slider','value')
)
def Question4(slider_value):

    #Scatter Plot
    fig=go.Figure(data=[go.Scatter(x=d4[d4['Month_End_Dummy'+
                                           str(slider_value)]!=1]['ON_Total_Volume']/d4[d4['Month_End_Dummy'+
                                                                                           str(slider_value)]!=1]['ON_N_repos'],
                                                                                           y=d4[d4['Month_End_Dummy'+str(slider_value)]!=1]['ON_Average_Rate'],
                                    mode = 'markers',name='non-MonthEnd',
                                    marker_color=d4[d4['Month_End_Dummy'+str(slider_value)]!=1]['Month_End_Dummy'+str(slider_value)].replace({1:'indianred',0:'teal'}),
                                    text=[i.date() for i in d4[d4['Month_End_Dummy'+str(slider_value)]!=1].index],hoverinfo = 'text'

                                    )])
    fig.add_trace(go.Scatter(x=d4[d4['Month_End_Dummy'+str(slider_value)]==1]['ON_Total_Volume']/d4[d4['Month_End_Dummy'+str(slider_value)]==1]['ON_N_repos'],
                             y=d4[d4['Month_End_Dummy'+str(slider_value)]==1]['ON_Average_Rate'],
                                        mode = 'markers',name='MonthEnd',
                                        marker_color=d4[d4['Month_End_Dummy'+str(slider_value)]==1]['Month_End_Dummy'+
                                                                                                    str(slider_value)].replace({1:'indianred',0:'teal'}),
                                        text=[i.date() for i in d4[d4['Month_End_Dummy'+str(slider_value)]==1].index],hoverinfo = 'text'
                                        ))

    fig.update_layout(template='plotly_dark',xaxis_title='Outstanding Volume',yaxis_title='Rate <b>p.a. (%)</b>',
                  plot_bgcolor= 'rgba(0, 0, 0, 0)',paper_bgcolor= 'rgba(0, 0, 0, 0)')



    return fig

@app.callback(
    Output('OLS_Table','data'),
    Output('QuanReg_Table','data'),
    Input('my-toggle-switch','value'),
    Input('my-slider','value'),
    Input('Tot-v-Avg','value')
    )
def dataset(InclDec,slider_value,outcome_var):
    d=d4.copy()
    print(InclDec)
    if InclDec==False:
        data=d[d.index.month!=12]
    else:
        data=d

    if outcome_var=='Total Outstanding Amount':
        OLSmodel=sm.OLS(data['ON_Total_Volume'],
                        sm.add_constant(data[['Month_End_Dummy'+str(slider_value),'ON_Average_Rate']])).fit(cov_type='HC1')
    else:
        OLSmodel=sm.OLS(data['ON_Total_Volume']/data['ON_N_repos'],
                        sm.add_constant(data[['Month_End_Dummy'+str(slider_value),'ON_Average_Rate']])).fit(cov_type='HC1')


    A=pd.concat([OLSmodel.params,OLSmodel.pvalues],axis=1).rename(columns={0:'Coefficient',1:'PValue'})

    print(outcome_var)

    if outcome_var!='Total Outstanding Amount':
        data['ON_Avg_Outstanding_Volume']=data['ON_Total_Volume']/data['ON_N_repos']
        quant_formula='ON_Avg_Outstanding_Volume ~ Month_End_Dummy'+str(slider_value)+' + ON_Average_Rate'
    else:
        quant_formula = 'ON_Total_Volume ~ Month_End_Dummy'+str(slider_value)+' + ON_Average_Rate'

    quant_mod = smf.quantreg(quant_formula, data)
    quantiles = [0.05,0.1, 0.25, 0.5, 0.75, 0.90,0.95]  # Quantiles of interest

    quant_results = []
    for quantile in quantiles:
        quant_result = quant_mod.fit(q=quantile, max_iter=1000)
        quant_results.append(quant_result)

    QuR=pd.DataFrame()
    for i in range(len(quantiles)):
      AA=pd.concat([quant_results[i].params, quant_results[i].pvalues],axis=1).rename(columns={0:'Coefficient',
                                                                                              1:'Pvalue'}).loc['Month_End_Dummy'+
                                                                                                               str(slider_value),:].rename(quantiles[i])
      QuR=pd.concat([QuR,AA],axis=1)


    return A.round(2).reset_index().to_dict('records'), QuR.round(2).T.reset_index().to_dict('records')

@app.callback(
    Output('Fixed_Effects','figure'),
    Input('Cash-Taker','value')
    )
def Question5(cashtak):
    print(cashtak)
    d=d5.copy()
    d['CashAmount_Demeaned_Std']=standardize(d['CashAmount']-d['CashAmount_Mean-per-Date'])
    d['Rate_Demeaned']=d['Rate']-d['Rate_Mean-per-Date']
    CashTaker_dummies = pd.get_dummies(d['CashTakerID'])
    X, Y= d[['CashAmount_Demeaned_Std']].join(CashTaker_dummies.drop(columns=[cashtak])), d[['Rate_Demeaned']]
    model=sm.OLS(Y, sm.add_constant(X)).fit(cov_type='HC1')
    Dummy_Coeff_CI=pd.concat( [model.params.iloc[2:].rename('CashTaker_Dummy_Coeff'),
            model.conf_int(alpha=0.02).rename(columns={0:'1%',1:'99%'}).iloc[2:,:]] ,axis=1)

    diff = [h - l for h, l in zip(Dummy_Coeff_CI['99%'], Dummy_Coeff_CI['1%'])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Dummy_Coeff_CI.index, y=Dummy_Coeff_CI['99%'], mode='markers',
                             name='UB(99%)', marker_color='#c92a52', marker_symbol='triangle-down', marker_size=15))
    fig.add_trace(go.Scatter(x=Dummy_Coeff_CI.index, y=Dummy_Coeff_CI['CashTaker_Dummy_Coeff'],
                             mode='markers', name='Coefficient', marker_color='teal', marker_line_width=1))
    fig.add_trace(go.Scatter(x=Dummy_Coeff_CI.index, y=Dummy_Coeff_CI['1%'], mode='markers',
                             name='LB (1%)', marker_color='#4d70c9', marker_symbol='triangle-up', marker_size=15))

    fig.add_trace(go.Bar(x=Dummy_Coeff_CI.index, y=diff, base=Dummy_Coeff_CI['1%'], width=0.01, marker_color='white', showlegend=False))
    fig.update_layout(template='plotly_dark',xaxis_title='Borrower ID',yaxis_title="Competitor's Differential Rate",
                      plot_bgcolor= 'rgba(0, 0, 0, 0)',paper_bgcolor= 'rgba(0, 0, 0, 0)')

    return fig


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)