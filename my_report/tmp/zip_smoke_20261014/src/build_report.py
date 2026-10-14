"""Build matched English/Chinese preliminary reports from verified experiment JSON.

Requires XeLaTeX to compile the generated .tex files. Content is deliberately
organised into nine body pages; no values are copied from the legacy report.
"""
import json
import re
from pathlib import Path
import numpy as np
from analyze import ROOT, LABEL, read_records
from config import ELECTRICITY_CONFIG, SOLAR_CONFIG
from data_loader import get_data_splits

def num(x, digits=3): return f'{x:,.{digits}f}'
def table(headers, rows, widths=None):
    spec=widths or ('l'+'r'*(len(headers)-1))
    return '\\begin{center}\\small\n\\begin{tabular}{'+spec+'}\\toprule\n'+' & '.join(headers)+r'\\\midrule'+'\n'+'\n'.join(' & '.join(map(str,r))+r'\\' for r in rows)+'\n\\bottomrule\\end{tabular}\\end{center}\n'
def figure(name, caption, height='5.5cm'):
    return r'\begin{center}\includegraphics[width=\textwidth,height='+height+r',keepaspectratio]{figures/'+name+r'.pdf}\captionof{figure}{'+caption+r'}\end{center}'
def sub(title,text): return '\\subsection{'+title+'}\n'+text+'\n'
def section(n,title,cn=False):
    author='本节撰写者：待填写。' if cn else 'Section author: to be completed.'
    return (r'\clearpage'+'\n' if n>1 else '')+'\\section{'+title+'}\n\\noindent\\textit{'+author+'}\n\n'

REFS=r'''
\begin{thebibliography}{9}\small
\bibitem{monash} R. Godahewa et al., ``Monash Time Series Forecasting Archive,'' NeurIPS Datasets and Benchmarks, 2021. \url{https://arxiv.org/abs/2105.06643}.
\bibitem{electricity} R. Godahewa et al., Electricity Dataset (Hourly Observations), Zenodo, record 4656140. \url{https://doi.org/10.5281/zenodo.4656140}.
\bibitem{solar} R. Godahewa et al., Solar Dataset (10 Minutes Observations), Zenodo, record 4656144. \url{https://doi.org/10.5281/zenodo.4656144}.
\bibitem{lstm} S. Hochreiter and J. Schmidhuber, ``Long Short-Term Memory,'' Neural Computation, 9(8), 1735--1780, 1997. \url{https://www.bioinf.jku.at/publications/older/2604.pdf}.
\bibitem{gru} K. Cho et al., ``Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation,'' EMNLP, 2014. \url{https://arxiv.org/abs/1406.1078}.
\bibitem{dlinear} A. Zeng et al., ``Are Transformers Effective for Time Series Forecasting?'' AAAI, 2023. \url{https://arxiv.org/abs/2205.13504}.
\bibitem{patch} Y. Nie et al., ``A Time Series is Worth 64 Words: Long-term Forecasting with Transformers,'' ICLR, 2023. \url{https://arxiv.org/abs/2211.14730}.
\bibitem{mase} R. J. Hyndman and A. B. Koehler, ``Another Look at Measures of Forecast Accuracy,'' International Journal of Forecasting, 22(4), 679--688, 2006. \url{https://robjhyndman.com/papers/mase.pdf}.
\bibitem{course} ELEN90099 teaching team, LDI Project Manual v1, Project Options v1, Inspiration Project v1, and supplied Prelim Report instructions/rubric, 2026. Local course materials.
\end{thebibliography}
'''

def build(cn=False):
    records=read_records(); summary=json.loads((ROOT/'results'/'summary.json').read_text())
    S={(r['dataset'],r['model']):r for r in summary}
    def r(ds,m):return S[ds,m]
    def ms(ds,m,key):
        a=r(ds,m);return num(a[key])+r' $\pm$ '+num(a[key+'_std'])
    def main_table(ds):
        return table(['Model','MAE','RMSE','MASE'],[[LABEL[m],ms(ds,m,'mae'),ms(ds,m,'rmse'),ms(ds,m,'mase')] for m in ['Persistence','DailySeasonalNaive','LSTM','GRU','CNN1D','CNNLSTM','DLinear','PatchTST']])
    ids={c.name:', '.join(c.channel_ids) for c in [ELECTRICITY_CONFIG,SOLAR_CONFIG]}
    splits={}
    for c in [ELECTRICITY_CONFIG,SOLAR_CONFIG]:
        ls,sc,raw,bounds=get_data_splits(c)
        splits[c.name]=dict(count=[len(ls[s+'_starts']) for s in ['train','val','test']],bounds=bounds,
            zero=float((raw[:bounds[1]]==0).mean())*100,raw=raw)
    gain=100*(1-r('Electricity','PatchTST')['mase']/r('Electricity','DailySeasonalNaive')['mase'])
    solar_penalty=100*(r('Solar','PatchTST')['mase']/r('Solar','DailySeasonalNaive')['mase']-1)
    env=records[0]['environment']
    e0=next(x for x in records if x['stage']=='main' and x['dataset']=='Electricity' and x['model']=='LSTM' and x['seed']==42)
    s0=next(x for x in records if x['stage']=='main' and x['dataset']=='Solar' and x['model']=='LSTM' and x['seed']==42)
    slice_rows=[]
    for ds in ['Electricity','Solar']:
        for m in ['DailySeasonalNaive','LSTM','DLinear','PatchTST']:
            slice_rows.append([ds if m=='DailySeasonalNaive' else '',LABEL[m],num(r(ds,m)['mae4']),num(r(ds,m)['slice_mae'])])
    resources=[]
    for m in ['LSTM','GRU','CNN1D','CNNLSTM','DLinear','PatchTST']:
        resources.append([LABEL[m],f"{r('Electricity',m)['params']:,}",f"{r('Solar',m)['params']:,}",
            num(r('Electricity',m)['seconds'],1)+' / '+num(r('Solar',m)['seconds'],1),
            num(r('Electricity',m)['latency_ms'],2)+' / '+num(r('Solar',m)['latency_ms'],2)])
    sens=[]
    for ds in ['Electricity','Solar']:
        for stage in ['window','fraction']:
            for model in ['LSTM','PatchTST']:
                a=[x for x in records if x['dataset']==ds and x['stage']==stage and x['model']==model]
                a.sort(key=lambda x:x['seq_len'] if stage=='window' else x['train_fraction'])
                sens.append([ds if stage=='window' and model=='LSTM' else '',
                    ('History' if stage=='window' else 'Data') if model=='LSTM' else '',LABEL[model]]+
                    [num(x['metrics_24h']['mase']) for x in a])
    def sensitivity_statement(ds,stage,model,cn=False):
        a=sorted([x for x in records if x['dataset']==ds and x['stage']==stage and x['model']==model],key=lambda x:x['seq_len'] if stage=='window' else x['train_fraction'])
        low,high=a[0],a[-1]
        if cn:return f"{ds} 的 {LABEL[model]} 在该因素从最低到最高配置下，MASE 由 {num(low['metrics_24h']['mase'])} 变为 {num(high['metrics_24h']['mase'])}。"
        return f"For {ds}, {LABEL[model]} changes from MASE {num(low['metrics_24h']['mase'])} to {num(high['metrics_24h']['mase'])} between the smallest and largest setting."
    pages=[]
    if not cn:
        pages.append(section(1,'Introduction, scope and literature')+r'''
\subsection{Engineering problem and motivation}
Day-ahead electricity demand and photovoltaic (PV) output forecasts can inform scheduling and storage decisions. Here the engineering objective is narrower: predict the next 24 hours using only the preceding 48 hours of observed target values, and assess whether added model complexity improves accuracy enough to justify its computational cost. Forecasting is studied as a supervised sequence-to-sequence regression problem, not as an operational dispatch controller. No claim about realised cost savings, grid safety or deployment is made.

The study uses two separate energy tasks: hourly Electricity and ten-minute Solar from the Monash archive~\cite{monash,electricity,solar}. Each task has 16 fixed, aligned channels and its own independently trained models. Weather, calendar features, future observations and other datasets are excluded. This limited information set makes comparisons clear, but also limits the ability to anticipate weather-driven changes in PV generation. The Solar release has a simulated source; using an original benchmark file does not make it a field deployment dataset~\cite{solar}.
\subsection{Research questions and measurable outcomes}
\begin{enumerate}
\item Do trainable models improve on persistence and a daily seasonal naive predictor on the same forecast origins?
\item How do recurrent memory, local convolution, hybrid encoding, linear decomposition and patch attention differ in accuracy and resource use?
\item Does the ranking change with forecast distance, high-load/active-output conditions, history length or available training data?
\end{enumerate}
Success is assessed by original-scale MAE and RMSE, channel-averaged seasonal MASE, and measured training/inference costs. Beating a strong daily baseline is more informative than beating persistence alone. A negative result is useful when the comparison protocol and its limits are explicit.
\subsection{Related methods and course connections}
LSTM~\cite{lstm} and GRU~\cite{gru} provide gated recurrent state updates; a temporal CNN tests local filters and weight sharing. Their comparison links recurrence, gradient-based learning and convolution to the course. CNN--LSTM tests a serial composition, while leaving open whether extra capacity rather than complementary mechanisms explains any improvement. DLinear~\cite{dlinear} challenges the assumption that nonlinear attention is necessary. PatchTST~\cite{patch} motivates patches and channel-shared encoding, giving a different temporal representation.

The implementation uses self-written PyTorch classes. In particular, the attention model is called \emph{PatchTST-style}: it preserves patching and channel independence but is not an exact reproduction of the published model or a direct import from Time-Series-Library. This report evaluates the supplied implementation, not a published benchmark ranking.
\subsection{Preliminary status and report structure}
This preliminary submission contains three-seed, five-epoch-budget main comparisons and single-seed sensitivity experiments on original releases. Sections 2--5 define the data, protocol, designs and implementation; Sections 6--8 report initial findings; Section 9 gives limitations, final-report tasks and conclusions. The complete final-report structure is therefore visible even where further validation remains planned~\cite{course}.
''')
        pages.append(section(2,'Data provenance and exploratory analysis')+r'''
\subsection{Release checks and frozen subset}
The downloaded ZIP and extracted TSF files are checksummed. Every source series is checked for finite values, common length/start time, and nonconstant training data. The selected IDs are matched by \texttt{series\_name}, never by an assumed zero-based numeric position. No interpolation or unit conversion is applied; quantities are reported in original release units because their physical interpretation has not been independently resolved for these aggregate files.
'''+table(['Property','Electricity','Solar'],[
        ['Source series / selected','321 / 16','137 / 16'],['Samples per series','26,304','52,560'],
        ['Sampling / daily period','1 hour / 24','10 minutes / 144'],['Source start','2012-01-01 00:00:01','2006-01-01 00:00:01'],
        ['History / horizon (steps)','48 / 24','288 / 144'],['Training zero values',num(splits['Electricity']['zero'],1)+r'\%',num(splits['Solar']['zero'],1)+r'\%']
        ])+r'''
Subset selection uses only the first 70\% of each source: stable sorting by mean, four rank groups, and four random draws per group with a newly initialised NumPy RNG (seed 42) for each dataset. The selection is independently reproduced and matches the frozen outline IDs. This stratifies average scale; it does not establish population representativeness.
\begin{quote}\small
Electricity: '''+ids['Electricity']+r'''.\\
Solar: '''+ids['Solar']+r'''.
\end{quote}
'''+figure('data_overview','Fixed first-channel profiles and training-only correlation. The first seven source days are displayed without selecting an attractive forecast.','6.1cm')+r'''
\subsection{Implications for model design}
The raw profiles and training correlations motivate daily baselines and both channel-mixing and channel-independent designs. Correlation alone does not prove a causal interaction or that cross-channel attention is needed. High solar zero prevalence can lower overall error while concealing errors during output activity; both full and conditional metrics are retained. Different users/plants have different scales, motivating per-channel normalisation and a scaled metric alongside absolute errors. The implicit source time grid is retained without assigning an unverified local timezone or interpreting clock hours as confirmed sunrise/sunset.
''')
        pages.append(section(3,'Forecasting formulation and evaluation protocol')+r'''
\subsection{Information available at each origin}
Let $X\in\mathbb R^{T\times C}$, $C=16$. At origin $t$, the input is $X_{t-L:t}$, using half-open slices, and the label is $Y_t=X_{t:t+H}$. Every model returns all $H\times C$ targets in one pass. Electricity uses $(L,H)=(48,24)$; Solar uses $(288,144)$. No target within $[t,t+H)$ enters the input. Later rolling origins may use observations revealed since earlier origins; this is rolling-origin forecasting rather than a single frozen forecast made at the beginning of the test interval.
\subsection{Chronological partition and standardisation}
Set $b_1=\lfloor0.70T\rfloor$ and $b_2=\lfloor0.80T\rfloor$. Training targets lie within $[0,b_1)$, validation within $[b_1,b_2)$, and test within $[b_2,T)$. A complete target must fit inside its own interval. Historical context may extend into an earlier split because it is already observed. Training origins advance by three hours; validation/test origins by one hour. Shuffling is limited to the already isolated training windows.
'''+table(['Dataset','$b_1$','$b_2$','Train origins','Val origins','Test origins'],[[ds,splits[ds]['bounds'][1],splits[ds]['bounds'][2],*splits[ds]['count']] for ds in ['Electricity','Solar']])+r'''
Each channel is standardised using only the available training interval: $z_{tc}=(x_{tc}-\mu_c)/\max(\sigma_c,10^{-8})$. Validation/test use the same statistics, and predictions are inverse transformed before reporting errors. Standardised MSE is used for fitting and validation selection. It gives a different channel weighting from raw-scale MAE; the distinction matters for interpreting rankings.
\subsection{Metrics and aggregation}
For $N$ origins, horizon $H$, channels $C$, and errors $e_{thc}=\hat y_{thc}-y_{thc}$,
\begin{align}
\mathrm{MAE}&=\frac1{NHC}\sum_{t,h,c}|e_{thc}|,&
\mathrm{RMSE}&=\sqrt{\frac1{NHC}\sum_{t,h,c}e_{thc}^2},\\
q_c&=\frac1{T_{\rm tr}-m}\sum_{s=m}^{T_{\rm tr}-1}|x_{sc}-x_{s-m,c}|,&
\mathrm{MASE}&=\frac1C\sum_c\frac{\mathrm{MAE}_c}{\max(q_c,10^{-8})}.
\end{align}
Here $m$ is one day and $q_c$ uses only that run's training interval~\cite{mase}. The global RMSE is not an average of channel RMSEs. MASE below one compares against in-training seasonal differences; it does not automatically imply beating the seasonal naive model on the test interval. MAE/RMSE from the two datasets must not be numerically ranked against each other.
\subsection{Forecast distance, conditional slices and dependence}
Report both the first four hours and the full 24 hours, plus each lead time in hours. The Electricity peak mask is true output at or above the channel's training 85th percentile. The Solar active mask is true output at or above 10\% of its training maximum. These are post-hoc evaluation masks, not inputs; active output is not a verified astronomical daytime label. Raw negative predictions remain in all main scores.

Hourly rolling windows overlap substantially. Their errors are dependent, so the number of windows is not an independent sample size. Seed standard deviations describe training variability only; no independent-window significance test is reported.
''')
        pages.append(section(4,'Model mechanisms and design choices')+r'''
\subsection{Two non-learning controls}
Persistence sets $\hat y_{t+h,c}=x_{t-1,c}$. Daily naive repeats the latest observed day, $\hat y_{t+h,c}=x_{t+h-m,c}$ for $h=0,\ldots,H-1$, where $H=m$. Both see only the same history as trainable models. They have no fitted parameters and test whether continuity or diurnal repetition already explains most predictable variation.
\subsection{Recurrent and convolutional networks}
The one-layer LSTM has hidden size 64 and updates its memory using
\[
c_s=f_s\odot c_{s-1}+i_s\odot\tanh(W_g[x_s,h_{s-1}]+b_g),\qquad h_s=o_s\odot\tanh(c_s),
\]
where $i_s,f_s,o_s$ are sigmoid gates~\cite{lstm}. The final hidden state is linearly projected to $HC$ outputs. Gating provides a learnable memory path; it does not guarantee that a short training budget learns a full weekly dependency. GRU uses update/reset gates and a single hidden state~\cite{gru}, reducing recurrent parameters. Both mix the 16 inputs at each time step and compress the history into a final state.

The 1D-CNN applies temporal kernels of width three, channels $16\to32\to64$, batch normalisation and ReLU, followed by adaptive average pooling to 16 positions and a dense $HC$ head. Its two convolutions have a five-step local receptive field before pooling. Pooling aggregates local features over the complete history; this is not a dilated temporal convolutional network and not proof of explicit long-range interaction. Symmetric convolution padding uses only the observed history, so it does not expose future targets.
\subsection{Hybrid CNN--LSTM}
The hybrid applies a $16\to64$ width-three convolution with batch normalisation/ReLU, then a 64-unit LSTM and direct dense head. The hypothesis is that local features simplify recurrent encoding. Comparing it with LSTM is a useful architecture comparison, but not an isolated causal ablation: its input representation and parameter count also change. Capacity-matched and frontend-removal controls remain final-report work.
\subsection{Decomposition and patch attention}
DLinear decomposes the history into an endpoint-replicated 25-step moving average $T$ and remainder $R=X-T$, and predicts
\[\hat Y_{c,:}=W_TT_{c,:}+W_RR_{c,:}+b_T+b_R.\]
The two maps are shared across channels and contain $2H(L+1)$ parameters. The remainder is not guaranteed to be a pure physical seasonal component. A 25-step smoother spans 25 hours on Electricity but only 4 h 10 min on Solar, so equal step settings are not equal physical filters~\cite{dlinear}.

The PatchTST-style model independently patches each channel (length 16, stride 8), projects patches to width 64, adds learnable positions, applies two Transformer encoder layers (four heads, feed-forward width 128, dropout 0.1), and flattens tokens to an $H$-output head. The scaled dot-product attention operation is $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$~\cite{patch}. Channels share weights after folding into the batch dimension; there is no explicit cross-channel attention. With aligned core windows, patch counts are 5 and 35, respectively. Patches span 16 hours versus 2 h 40 min, another frequency-dependent design difference.

This custom attention implementation uses PyTorch LayerNorm encoder blocks, learnable zero-initialised positions, and global training-set scaling without RevIN. It is therefore a simplified implementation inspired by PatchTST, not an exact upstream reproduction. DLinear also uses the local class's random initialisation. These choices are recorded rather than attributing the observations to all versions of the original models.
''')
        pages.append(section(5,'Implementation, training and reproducibility')+r'''
\subsection{Initial experiment budget and validation selection}
The preliminary budget is fixed at five maximum epochs, patience three, batch size 32, Adam with learning rate $10^{-3}$ and weight decay $10^{-4}$. No model-specific test-driven search is performed. Main trainable comparisons use seeds 42, 43 and 44; each sensitivity setting uses seed 42. This is a deliberately limited pilot, not a claim of convergence. The final plan extends the maximum to 20 epochs with validation-based choices.

Each epoch visits the complete training-origin set. Validation MSE is summed over all target elements and divided by their count, so a smaller final batch is weighted correctly. The lowest validation-loss state is retained; test metrics are computed from that state even when it is not the final epoch. Identical epoch limits do not equalise parameter counts, FLOPs, optimiser steps across data fractions, or wall-clock budgets. These differences are logged rather than hidden.
\subsection{Measured environment and resources}
'''+f"All new runs use {env['device_name']}, PyTorch {env['torch']}, Python {env['python']} and NumPy {env['numpy']}. "+r'''
This is a local GPU experiment, not measured Colab T4 performance. The same code can be run in Colab; elapsed times must be measured again there.

Training time includes the epoch loop and validation. Peak allocated CUDA memory is recorded for the training phase. Inference measures a model-only forward pass with the input already resident on the device, after ten warmups and 30 synchronised wall-clock repetitions. It excludes loading, host-device transfers, standardisation and database/file writes; it is not an end-to-end deployment SLA. The resource summary below averages training times and per-run median latencies over main seeds.\par
'''+table(['Model','Params E','Params S','Train s E / S','Median ms E / S'],resources)+r'''
\subsection{What the code package records}
\texttt{prepare\_data.py} obtains original releases and saves hashes/IDs; \texttt{data\_loader.py} enforces verified provenance and split rules. \texttt{experiment.py} trains and evaluates every model through one interface. Unique run directories hold the best checkpoint, scaler, split origins and all test predictions. JSON records and the SQLite \texttt{runs} table retain hyperparameters, losses, epochs, optimiser steps, environment, metrics and relative artifact paths. SQLite satisfies database storage; vector retrieval is not implemented or claimed.

\texttt{verify\_results.py --reload} recomputes errors from saved predictions, checks source alignment and database consistency, and reloads each checkpoint on CPU for a sampled forecast comparison. Contract tests cover leakage boundaries, provenance failures, metrics, forward/backward passes and channel strategy. Successful software tests establish implementation checks, not forecast generalisation. The README documents reconstruction, rerunning, analysis and report compilation.
''')
        pages.append(section(6,'Initial results and comparative analysis')+r'''
The tables report mean $\pm$ sample standard deviation over seeds 42, 43 and 44 on the same test origins. Deterministic baselines have identical repeated predictions and effectively zero seed deviation; their three records are not three independent forecasting trials. All errors use original release scale except dimensionless MASE. Tables and figures are generated from the new run records, not the legacy database.
\subsection{Electricity: daily structure is a strong control}
'''+main_table('Electricity')+f"The PatchTST-style model has mean seasonal MASE {num(r('Electricity','PatchTST')['mase'])}, compared with {num(r('Electricity','DailySeasonalNaive')['mase'])} for daily naive: an observed {num(gain,1)}\\% reduction. "+r'''
The result supports further study of patch-based temporal encoding on this subset, not a general superiority claim. Large raw-scale channels can dominate MAE/RMSE while channel-scaled MASE rewards relative improvement across channels. The comparison should therefore retain all three metrics. Recurrent or hybrid underperformance after five epochs may reflect optimisation, representation bottlenecks or the information set, rather than proving those model families unsuitable.
\subsection{Solar: learn the daily baseline before adding complexity}
'''+main_table('Solar')+f"Daily naive has mean MASE {num(r('Solar','DailySeasonalNaive')['mase'])}; the PatchTST-style model is {num(solar_penalty,1)}\\% higher. "+r'''
A strong daily baseline is consistent with repeated output profiles, but does not establish that clouds are predictable or that neural models cannot improve with adequate validation-based training. A linear or attention model may smooth sharp output peaks; a recurrent final-state bottleneck also faces a much larger 144-step output. The experiment contains no weather covariates to anticipate genuinely novel cloud conditions.
'''+figure('comparison','Initial main comparisons; error bars show training-seed variability, not confidence intervals over independent test days.','3.7cm'))
        pages.append(section(7,'Horizon, conditional errors and difficult cases')+r'''
\subsection{Lead time should be read in physical hours}
'''+figure('lead_time','Mean per-lead MAE across main seeds; persistence omitted for legibility. Solar steps are ten minutes; Electricity steps are one hour.','4.3cm')+r'''
Direct multi-output models do not recursively feed their own predictions back. Their lead-time errors may still vary because each output has a different relation to the input and because predictable daily structure reappears. A nonmonotonic curve is therefore possible; monotonically growing error should not be assumed. Persistence loses the benefit of phase alignment, whereas daily naive directly preserves it. These curves describe this rolling-origin evaluation, not an independent error-growth experiment.
'''+table(['Dataset','Model','First 4h MAE','Conditional MAE'],slice_rows)+r'''
The conditional column uses the training-defined high-load mask for Electricity and active-output mask for Solar. The masks are identical across models within a dataset/configuration. Conditioning can change the mix of channels and magnitudes as well as difficulty; a higher conditional MAE is not alone evidence of a distinct physical failure mechanism. Mask prevalence is saved in the run records.
\subsection{A fixed hard-case rule, including imperfect forecasts}
'''+figure('hard_cases','For the first frozen channel, select the origin containing the largest true test target; ties use the earliest origin. The rule never selects by a model error. Seed 42 is fixed, not chosen as the best seed.','4.0cm')+r'''
These selected peak-output cases expose amplitude and phase mismatches that averaged errors can hide. They are stress illustrations rather than representative or independent evidence. Negative raw outputs are retained in the scores; universal nonnegative clipping can be assessed later as a separate post-processing comparison. The supplied case-selection JSON records the channel and source index so the plots are reproducible.
''')
        pages.append(section(8,'History and training-data sensitivity')+r'''
\subsection{Controlled design}
Two complementary experiments compare LSTM and PatchTST-style with seed 42 and the same five-epoch ceiling. History lengths are 24, 48 and 168 hours: Electricity lengths 24/48/168; Solar 144/288/1008. All lengths share the training origins admissible under the maximum length, and the same validation/test origins. Thus the 48-hour sensitivity run is retrained on a different common training-origin set than the main comparison; it is not replaced with the main score.

For data amount, use the latest 25\%, 50\% and 100\% of the original training interval, always ending at $b_1$. Both input and target stay within the retained training segment. Each fraction refits its own training-only scaler; fixed channel IDs and validation/test regions do not change. Model selection remains validation-based. The 100\% data setting is rerun rather than borrowed from a different budget.
'''+table(['Dataset','Factor','Model','Small','Middle','Large'],sens)+r'''
\noindent\small History columns: 24/48/168 hours. Data columns: 25/50/100\%. Values are seasonal MASE.\normalsize
'''+figure('sensitivity','Single-seed controlled sensitivity; lines connect tested settings and do not imply interpolated performance or statistical significance.','5.7cm')+r'''
\subsection{What can and cannot be inferred}
'''+sensitivity_statement('Electricity','window','PatchTST')+' '+sensitivity_statement('Solar','window','LSTM')+' '+sensitivity_statement('Electricity','fraction','LSTM')+r'''
 These endpoint comparisons do not establish monotonic benefit; the middle setting must also be read. Longer history changes PatchTST-style patch count and dense-head capacity, and changes computation even when recurrent parameter count stays constant. It is therefore a history-plus-capacity sensitivity, not an isolated memory mechanism test.

The fraction experiment changes season coverage, data recency composition and the number of optimiser updates within five epochs. Its MASE denominator is also recomputed on each available segment, so cross-fraction MASE changes can reflect both forecast errors and the reference scale. Use original-scale MAE/RMSE alongside MASE (stored in each JSON), and do not attribute all differences to sample count alone. Multiseed repetitions and a fixed-update-budget comparison remain final-report tasks.
''')
        pages.append(section(9,'Engineering judgement, limitations and final-report plan')+r'''
\subsection{A provisional engineering choice}
The relevant choice is an accuracy--cost trade-off on each task, not a single architecture winner. On Electricity, compare the PatchTST-style gain with a zero-parameter daily baseline and DLinear's compact maps. On Solar, daily naive is a strong current reference and a sensible fallback under this restricted information set. A more complex model should earn its additional cost through validation that matters for the application, such as peak or active-output errors. None of these initial error reductions is converted into monetary savings or storage control benefit.

Parameter count measures stored weights, while peak memory also includes activations/workspaces. Neither alone measures embedded feasibility: operators, numerical precision, runtime support and end-to-end latency must be checked on the actual target. The observed laptop GPU timings do not support a T4 throughput claim or a microcontroller deployment claim. A Pareto judgement is conditional on chosen objectives and these measurements, not a universal optimum.
\subsection{Limits that affect interpretation}
\begin{enumerate}
\item \textbf{Scope:} the 16 selected channels and one chronological test interval do not establish generalisation to all users/plants, another year, or another region. Solar has a simulated source and no verified astronomical/timezone labels.
\item \textbf{Training:} five epochs and fixed common settings provide a pilot, not equally optimised architectures. Three seeds measure random-initialisation/training variability, not all sources of uncertainty. Some validation curves still improve at the budget limit.
\item \textbf{Mechanisms:} architecture comparisons are confounded by capacity, physical filter span, normalisation and input representation. The self-written PatchTST-style class is not an exact published implementation; pretrained/foundation models and external regressors are outside current scope.
\item \textbf{Evaluation:} overlapping windows and repeated targets induce dependence. Training-scaled MASE, raw MAE/RMSE and conditional slices answer different weighting questions. The test period was already examined in earlier project iterations, so it is not an untouched external blind holdout. New settings were fixed rather than selected by test ranking.
\end{enumerate}
\subsection{Concrete path to the final report}
\begin{enumerate}
\item \textbf{Validation and budget:} inspect losses, extend the ceiling to 20 epochs, and permit a small, documented validation-only search. Record all attempted configurations and equalise the tuning budget; never select the best test seed.
\item \textbf{Sensitivity and ablation:} repeat each sensitivity setting across seeds, report MAE/RMSE as well as MASE, match update budgets as an additional control, and add a capacity-matched CNN frontend comparison. Retain the frozen channel subset and common origins.
\item \textbf{Error analysis:} add channel-level results, weekday/weekend analysis only after timestamp semantics are checked, more predefined hard scenes, residual dependence, and a uniform clipping comparison. If statistical inference is needed, use time blocks rather than treating overlapping windows as independent.
\item \textbf{Deliverables:} confirm authors and section responsibility, polish the final narrative within its separate 20-page total limit, retain the code/database evidence, and prepare the course Q\&A and at-most-three-minute video~\cite{course}.
\end{enumerate}
\subsection{Preliminary conclusion}
The verified experiment shows why baseline strength and protocol correctness precede model complexity. Patch-based encoding merits further Electricity validation, while Solar exposes a substantial gap to the daily control under the pilot budget. The contribution at this stage is a reproducible, comparable set of initial results, an explicit account of model mechanisms, and a controlled plan to test whether the observed trade-offs survive fuller training and broader validation.
''')
    else:
        pages=chinese_pages(section,table,figure,main_table,resources,sens,slice_rows,ids,splits,env,r,num,gain,solar_penalty,sensitivity_statement)
    preamble=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=20mm,headheight=15pt]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,caption,enumitem,array,fancyhdr}
\usepackage{fontspec}
'''+ ('\\usepackage[UTF8,fontset=windows]{ctex}\n' if cn else '')+r'''
\setmainfont{Latin Modern Roman}
\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue]{hyperref}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\setlist{nosep,leftmargin=1.5em,topsep=3pt}
\captionsetup{font=small,labelfont=bf,skip=4pt}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{ELEN90099 | LDI Preliminary Report}\fancyhead[R]{October 2026}\fancyfoot[C]{\thepage}
\setlength{\emergencystretch}{3em}
\begin{document}\begin{titlepage}\thispagestyle{empty}\centering\vspace*{22mm}
{\Large ELEN90099 Deep Learning\par}\vspace{8mm}
{\LARGE\bfseries '''+(r'用电负荷与光伏出力的多步预测\\[4mm]初步实验、模型比较与验证计划' if cn else r'Multi-step Electricity Load and Solar Output Forecasting\\[4mm]Initial Experiments, Model Comparisons and Validation Plan')+r'''\par}\vspace{10mm}
{\large Learn--Design--Implement Project\\Preliminary Report / '''+('中文对照版' if cn else 'English Submission Version')+r'''\par}\vspace{12mm}
'''+(r'两位作者英文姓名与学号：待填写。\\[3mm]各节撰写责任：待填写。' if cn else r'Both student names and student IDs: TO BE COMPLETED.\\[3mm]Section authorship: TO BE COMPLETED.')+r'''
\vspace{12mm}\begin{minipage}{0.88\textwidth}
\textbf{'''+('摘要。' if cn else 'Abstract. ')+('本项目以原始 Monash 发布文件的固定16通道子集为对象，比较历史48小时到未来24小时的用电负荷和光伏出力预测。所有模型采用相同时间划分、训练区间标准化和评价起点；主实验使用三个固定种子、每次最多五轮训练，另进行单种子的历史窗口和训练数据量敏感性研究。模型覆盖持久化、日周期朴素基线、LSTM、GRU、卷积、CNN–LSTM、自写DLinear和简化PatchTST风格模型。初步结果显示，Electricity上patch模型具有进一步验证价值，而Solar上日周期基线更强。报告区分正式发布数据与本地合成信号、实测成本与部署推断，提供可复算预测、权重和SQLite实验记录，并明确终稿验证计划。' if cn else 'Using fixed 16-channel subsets from original Monash releases, this project compares 48-hour-history to 24-hour-horizon forecasting for electricity load and solar output. All models share chronological splits, training-only scaling and evaluation origins. Initial main experiments use three fixed seeds and a five-epoch ceiling, with single-seed history/data sensitivity studies. The methods span persistence, daily naive, LSTM, GRU, CNN, CNN--LSTM, a self-written DLinear and a simplified PatchTST-style model. The patch model merits further Electricity validation, whereas daily naive remains stronger on Solar under this budget. Saved predictions, checkpoints and SQLite records support result verification. The report distinguishes observed accuracy and local GPU costs from untested deployment implications, and gives a concrete final-report plan.')+r'''}
\end{minipage}\vfill
'''+(r'正文9页；封面和参考文献不计入初步报告正文页数。\\作者信息按当前约定保留待填写。' if cn else r'Nine body pages; title and references excluded from the preliminary page count.\\Author fields intentionally remain incomplete pending confirmation.')+r'''
\end{titlepage}\setcounter{page}{1}
'''
    preamble=re.sub(r'\\textbf\{(Abstract\. |摘要。)(.*?)\}\n\\end\{minipage\}',r'\\textbf{\1}\2\n\\end{minipage}',preamble,flags=re.S)
    path=ROOT/'report'/('LDI_Preliminary_Report_CN.tex' if cn else 'LDI_Preliminary_Report_EN.tex')
    path.write_text(preamble+'\n'.join(pages)+r'\clearpage'+REFS+r'\end{document}',encoding='utf-8')
    return path

def chinese_pages(section,table,figure,main_table,resources,sens,slice_rows,ids,splits,env,r,num,gain,solar_penalty,sensitivity_statement):
    # Parallel narrative: same equations, data, figures and generated numbers.
    p=[]
    p.append(section(1,'引言、研究范围与相关工作',True)+r'''
\subsection{工程问题与动机}
未来一天的用电负荷与光伏出力预测，可为调度和储能决策提供信息。本项目将目标限定为：仅凭过去48小时的目标观测，一次预测未来24小时，并比较增加模型复杂度能否带来值得其计算代价的误差改善。本项目研究监督式序列回归，不构建实际调度控制器，也不把预测误差直接换算为经济收益、电网安全改善或已部署效果。

研究采用Monash时序档案的Electricity小时级和Solar十分钟级两个独立任务\cite{monash,electricity,solar}。每个任务固定16条对齐序列，分别训练模型；不加入天气、日历、未来观测或跨数据集信息。这使信息条件更明确，同时限制模型预测突发天气变化的能力。Solar原始来源包含模拟数据；使用正式发布文件，不等于数据已成为真实电站部署记录\cite{solar}。
\subsection{研究问题与成功标准}
\begin{enumerate}
\item 在相同预测起点上，可训练模型是否优于持久化与日周期朴素基线？
\item 循环记忆、局部卷积、混合编码、线性分解和patch注意力，在误差和资源需求上有何差别？
\item 排名是否随预测距离、高负荷/活跃出力条件、历史长度和可用训练数据改变？
\end{enumerate}
评价采用原尺度MAE、RMSE、通道平均季节性MASE，以及实测训练和推理成本。超过较强的日周期基线，比只超过持久化更有说服力。只要协议明确、局限透明，复杂模型没有改善也可以构成有意义的初步结论。
\subsection{文献与课程知识的连接}
LSTM\cite{lstm}和GRU\cite{gru}通过门控更新状态；时间卷积用于检验局部滤波和权重共享；这些比较把递归计算、梯度学习和卷积机制落到同一预测任务。CNN–LSTM检验串联特征提取，但不预设其改善必然来自模块互补。DLinear\cite{dlinear}为复杂非线性与注意力是否必要提供反例参照；PatchTST\cite{patch}启发时间分块与跨通道共享编码，构成不同的时间表示。

当前模型均为自写PyTorch实现。注意力模型明确称为“PatchTST风格”：保留patch和通道独立思想，但不是原论文完整复现，也未直接导入Time-Series-Library。因此报告评价的是随附实现，不能把结果归为原论文官方排名。
\subsection{初步进度与终稿结构}
本次初步报告包含正式发布数据上的三种子主实验，以及单种子敏感性比较；可训练模型每次最多五轮。第2–5节给出数据、协议、设计和实现，第6–8节报告初步结果，第9节讨论局限、终稿任务和结论。终稿章节结构已完整呈现，未完成的进一步验证明确作为后续任务\cite{course}。两位作者与每节撰写责任按当前约定保留待填写，不沿用旧稿的虚构学号。
''')
    p.append(section(2,'数据来源核验与探索性分析',True)+r'''
\subsection{发布文件与固定子集}
对下载ZIP和解压TSF保存SHA256；检查全部原始序列的有限性、共同长度/起点及训练区间非常数性。通过\texttt{series\_name}精确匹配通道，不将T5等名称当成数组索引。不插值、不额外转换单位；聚合发布版本的物理单位尚未独立确认，因此统一使用“原始发布单位”。
'''+table(['属性','Electricity','Solar'],[['原序列数 / 使用数','321 / 16','137 / 16'],['每条长度','26,304','52,560'],['采样 / 日周期','1小时 / 24','10分钟 / 144'],['TSF起点','2012-01-01 00:00:01','2006-01-01 00:00:01'],['历史 / 预测步数','48 / 24','288 / 144'],['训练区间零值比例',num(splits['Electricity']['zero'],1)+r'\%',num(splits['Solar']['zero'],1)+r'\%']])+r'''
选择仅使用前70\%区间：按均值稳定排序，分成四组，每组无放回抽四条；两个数据集分别重建seed42的NumPy随机数生成器。此次独立复现的选择与大纲固定名单一致。这覆盖不同平均水平，不证明子集代表总体。
\begin{quote}\small Electricity: '''+ids['Electricity']+r'''.\\Solar: '''+ids['Solar']+r'''.\end{quote}
'''+figure('data_overview','固定首通道的前七天曲线与训练区间相关性；没有按预测好坏选择数据片段。','6.1cm')+r'''
\subsection{对设计的启示与解释边界}
训练区间中的日内结构为日周期基线提供动机；相关性支持比较混合通道与独立通道模型，但不证明因果关系，也不能据此断言跨通道注意力必要。Solar大量零值可能压低总体误差，因此同时保留全时段和活跃出力条件指标。对象尺度不同，既需要逐通道标准化，也需要保留尺度归一化与原尺度两类评价。

两套数据分别来自不同年份和地区，不能拼接为对应负荷与电站。不擅自附加澳大利亚时区；发布文件中的共同起点和间隔仅支持格式对齐，不证明夏令时、当地时间或测量机制都经过验证。Solar的模拟来源还限制了结论向实际电站推广。
''')
    p.append(section(3,'预测问题定义与评价协议',True)+r'''
\subsection{每个预测起点的信息边界}
设$X\in\mathbb R^{T\times C}$，$C=16$。起点$t$的输入是半开区间$X_{t-L:t}$，目标为$Y_t=X_{t:t+H}$。所有模型一次输出全部$H\times C$目标；Electricity的$(L,H)=(48,24)$，Solar为$(288,144)$。未来标签不进入历史输入。后续滚动起点可以使用此前已揭示的观测；这属于滚动起点评价，不是在测试区间开始时一次冻结整段信息。
\subsection{时间划分与标准化}
取$b_1=\lfloor0.70T\rfloor$、$b_2=\lfloor0.80T\rfloor$。训练、验证和测试目标分别完全位于$[0,b_1)$、$[b_1,b_2)$和$[b_2,T)$，目标窗口不能跨越其所属区间。历史上下文可来自更早分区，因为在预测时已经观测到。训练起点间隔3小时，验证与测试间隔1小时。只打乱已经隔离好的训练窗口，不随机划分重叠窗口。
'''+table(['数据集','$b_1$','$b_2$','训练窗口','验证窗口','测试窗口'],[[ds,splits[ds]['bounds'][1],splits[ds]['bounds'][2],*splits[ds]['count']] for ds in ['Electricity','Solar']])+r'''
各通道仅以当前可用训练片段拟合$z_{tc}=(x_{tc}-\mu_c)/\max(\sigma_c,10^{-8})$；验证和测试使用相同统计量。预测反变换后才计算报告指标。训练与最佳权重选择使用标准化MSE，其通道权重和原尺度MAE不同，不能把训练损失排名直接等同于工程误差排名。
\subsection{指标与汇总方式}
设$N$个起点、预测跨度$H$、通道$C$，误差$e_{thc}=\hat y_{thc}-y_{thc}$，则
\begin{align}
\mathrm{MAE}&=\frac1{NHC}\sum_{t,h,c}|e_{thc}|,&\mathrm{RMSE}&=\sqrt{\frac1{NHC}\sum_{t,h,c}e_{thc}^2},\\
q_c&=\frac1{T_{\rm tr}-m}\sum_{s=m}^{T_{\rm tr}-1}|x_{sc}-x_{s-m,c}|,&\mathrm{MASE}&=\frac1C\sum_c\frac{\mathrm{MAE}_c}{\max(q_c,10^{-8})}.
\end{align}
其中$m$为一天，分母$q_c$仅使用该次训练区间\cite{mase}。整体RMSE不是通道RMSE的简单平均。MASE小于1表示优于训练区间的季节差分尺度，不自动意味着优于测试区间的日周期朴素预测。两套数据的MAE/RMSE单位及尺度不同，不能直接跨数据集比较数值大小。
\subsection{预测距离、条件切片与依赖性}
分别报告前4小时、完整24小时，以及按实际小时的逐距离误差。Electricity高负荷条件为真实值达到通道训练区间85\%分位数；Solar活跃条件为达到训练最大值的10\%。它们是事后评价条件，不是模型输入，也不是已确认的天文白天。主要结果保留所有原始负预测，不通过选择性截断改变成绩。

每小时滚动预测导致标签反复出现，窗口误差具有时间依赖。窗口数量不是独立样本数量；种子标准差仅描述训练随机性，因此不报告基于独立窗口假设的显著性结论。
''')
    p.append(section(4,'模型机制与设计理由',True)+r'''
\subsection{无需训练的两个对照}
持久化令$\hat y_{t+h,c}=x_{t-1,c}$；日周期朴素预测令$\hat y_{t+h,c}=x_{t+h-m,c}$，其中$h=0,\ldots,H-1$且$H=m$。两者与神经网络接触相同历史，没有拟合参数，用于检验连续性与日内重复是否已解释大部分可预测变化。
\subsection{循环网络与卷积网络}
单层LSTM隐藏维64，利用门控记忆更新
\[
c_s=f_s\odot c_{s-1}+i_s\odot\tanh(W_g[x_s,h_{s-1}]+b_g),\qquad h_s=o_s\odot\tanh(c_s),
\]
其中$i_s,f_s,o_s$是sigmoid门\cite{lstm}。最后隐藏状态线性映射到$HC$个输出。门控提供可学习的记忆通路，但不保证五轮训练能学到周依赖。GRU通过更新门与重置门维护单隐藏状态\cite{gru}，减少循环参数。两者在每步混合16通道，并将整个历史压入末状态。

1D-CNN采用两层宽度3卷积，通道$16\to32\to64$，搭配BatchNorm与ReLU，自适应平均池化到16个位置，再由全连接层输出$HC$。池化前两层卷积的局部感受野为5步，池化汇总全历史的局部特征；这不是扩张TCN，也不证明它显式建模了远距离交互。对称padding仅在已观测历史内部运算，不泄漏未来目标。
\subsection{CNN–LSTM混合结构}
先用$16\to64$、宽度3卷积和BatchNorm/ReLU提取特征，再送入64维LSTM和直接预测头。设计假设是局部模式可简化循环编码。与LSTM比较属于架构比较，尚不能称为独立因果消融：输入表示和参数量也改变。容量匹配、移除前端等进一步控制留到终稿。
\subsection{线性分解与patch注意力}
DLinear以端点复制的25步移动平均得到趋势$T$和余项$R=X-T$，预测
\[\hat Y_{c,:}=W_TT_{c,:}+W_RR_{c,:}+b_T+b_R.\]
双分支映射在通道间共享，总参数$2H(L+1)$。余项不保证是纯物理季节分量。同样25步在Electricity覆盖25小时，在Solar仅4小时10分，因此相同步数不是相同物理滤波尺度\cite{dlinear}。

PatchTST风格模型逐通道分块，patch长度16、步幅8；映射到64维并加入可学习位置编码，经过两层Transformer（4头、前馈128、dropout0.1），展平后输出未来$H$步。注意力为$\mathrm{softmax}(QK^\top/\sqrt{d_k})V$\cite{patch}。通道折叠到batch维共享权重，不存在显式跨通道注意力。核心窗口的patch数量分别为5和35；patch物理长度为16小时与2小时40分，再次体现采样率影响。

当前自写实现采用PyTorch LayerNorm编码块、零初始化可学习位置向量和训练集整体标准化，没有RevIN；不能称为官方PatchTST精确复现。DLinear同样采用本地类的随机初始化。报告保留这些差别，不把本实现的观测推广为原模型所有实现的共同性质。
''')
    p.append(section(5,'实现、训练与可复现性',True)+r'''
\subsection{初步预算与验证集选择}
初步配置预先固定为最多5轮、patience=3、batch=32、Adam学习率$10^{-3}$、weight decay=$10^{-4}$。主实验采用42、43、44三个种子，敏感性采用seed42；不依据测试排名进行特定模型搜索。这是明确受限的初步实验，不是收敛保证；终稿计划增加到最多20轮并通过验证集选择配置。

每轮遍历完整训练起点。验证MSE按所有目标元素累计后除以元素数量，避免末尾小batch获得过高权重。保留验证损失最低的状态，测试使用该状态而不一定是最后一轮。相同epoch上限并不等于参数、FLOPs、数据缩减时的optimizer steps或实际时间相同；这些量均记录。
\subsection{实测环境与资源}
'''+f"新版运行设备为{env['device_name']}，PyTorch {env['torch']}，Python {env['python']}，NumPy {env['numpy']}。"+r'''
这是本地GPU结果，不是Colab T4实测。代码可在Colab运行，但应重新测量资源成本。

训练耗时包含训练轮次及验证，训练峰值采用CUDA allocated memory。推理为输入已驻留设备的模型前向，预热10次后同步计时30次；不含数据加载、传输、标准化和数据库/文件写入，不是端到端部署延迟保证。下表汇总主实验种子平均训练时间，以及各次推理中位数的平均；E/S分别代表Electricity/Solar。
'''+table(['模型','参数 E','参数 S','训练秒 E / S','中位毫秒 E / S'],resources)+r'''
\subsection{代码交付记录什么}
\texttt{prepare\_data.py}获取正式文件并保存哈希和ID；\texttt{data\_loader.py}验证数据来源及切分规则；\texttt{experiment.py}统一训练与评估。每次运行采用唯一目录，保存最佳权重、scaler、全部分区起点和完整测试预测。JSON与SQLite的\texttt{runs}表保留超参数、曲线、epoch、优化步数、环境、指标与相对路径。已实现数据库存储，但不宣称实现了向量检索。

\texttt{verify\_results.py --reload}从预测文件复算误差，检查源数据对齐、数据库一致性，并在CPU重载每个权重进行抽样预测对照。测试覆盖泄漏边界、来源缺失/篡改、指标、前后向梯度与通道策略。通过这些检查说明实现合同得到验证，不等于证明模型泛化。README提供获取数据、运行、重算、分析和编译步骤。

主结果均来自独立的新目录。旧稿的本地合成缓存、CPU数据库和标注不一致的T4图没有进入本报告；独立核验说明保留具体证据，避免把能够复算错误数据上的数字误当成正确的数据来源。
''')
    p.append(section(6,'初步结果与比较分析',True)+r'''
下表为三个固定种子的均值$\pm$样本标准差。确定性基线的预测相同、种子标准差近零，三份记录不算三个独立预测试验。MAE/RMSE保留原发布尺度，MASE无量纲。表图直接由新版运行记录生成，不引用历史数据库成绩。
\subsection{Electricity：日周期是较强对照}
'''+main_table('Electricity')+f"PatchTST风格模型的平均MASE为{num(r('Electricity','PatchTST')['mase'])}，日周期基线为{num(r('Electricity','DailySeasonalNaive')['mase'])}，对应观测到的{num(gain,1)}\\%下降。"+r'''
该结果支持继续研究此子集上的patch时间编码，不代表普遍优势。原尺度MAE/RMSE受大通道影响更强，通道缩放MASE强调各通道相对误差，因此应同时阅读。循环和混合模型在五轮后的不足，可能来自优化不足、表示压缩或信息条件，不能证明这些架构不适用。
\subsection{Solar：先超过日周期基线，再讨论复杂度收益}
'''+main_table('Solar')+f"日周期基线平均MASE为{num(r('Solar','DailySeasonalNaive')['mase'])}；PatchTST风格模型比它高{num(solar_penalty,1)}\\%。"+r'''
较强的日周期基线与重复出力形状一致，但不能据此证明云层可预测，也不能证明神经网络充分训练后无法改善。线性或注意力映射可能平滑尖峰，循环末状态还需输出较长的144步；当前没有天气变量来预判新云层变化。
'''+figure('comparison','主实验初步比较；误差棒表示训练种子差异，不是独立测试日的置信区间。','3.7cm'))
    p.append(section(7,'预测距离、条件误差与困难案例',True)+r'''
\subsection{以实际小时解释预测距离}
'''+figure('lead_time','三个主实验种子的逐距离平均MAE；为便于阅读省略持久化。Solar每步10分钟，Electricity每步1小时。','4.3cm')+r'''
直接多步模型不将自身预测递归回灌，但不同输出与输入的关系、周期结构再次出现，都可使误差随距离变化。因此误差曲线不必单调增长。持久化没有对齐未来日内相位，日周期基线则直接保留相位。曲线描述本次滚动起点评价，不是独立的误差累积因果实验。
'''+table(['数据集','模型','前4小时MAE','条件MAE'],slice_rows)+r'''
条件列对应Electricity训练分位数定义的高负荷，或Solar训练最大值定义的活跃出力；同数据集/配置内各模型使用相同掩码。条件评价也改变通道和幅值构成，条件MAE较高本身不能证明特定物理失败机制。筛选比例保存在运行记录中。
\subsection{预先固定的困难场景规则}
'''+figure('hard_cases','固定首通道，选择包含测试真实最大目标值的最早起点；不依模型误差选例，固定seed42而非最好种子。','4.0cm')+r'''
这些峰值案例展示平均误差可能掩盖的幅值与相位偏差，但不是代表性或独立的总体证据。全部原始负预测仍计入主要分数；后续可另行比较所有模型统一非负截断的效果。\texttt{case\_selection.json}记录通道与源索引，能复现选择过程。
''')
    p.append(section(8,'历史长度与训练数据量敏感性',True)+r'''
\subsection{受控实验设计}
LSTM和PatchTST风格模型均用seed42、最多5轮。历史取24/48/168小时：Electricity为24/48/168步，Solar为144/288/1008步。全部长度使用最大历史允许的共同训练起点，以及相同验证/测试起点。因此敏感性中的48小时运行是在较少的共同训练起点上重训，不能拿主实验成绩直接代替。

数据量实验保留原训练区间末尾25\%、50\%、100\%，结束时间固定$b_1$；输入与标签均位于保留片段内。各比例单独拟合可用训练scaler，固定通道和验证/测试区间不变，继续由验证损失选择最佳权重。100\%配置也重新运行，避免混入其他预算的结果。
'''+table(['数据集','因素','模型','小','中','大'],sens)+r'''
\noindent\small History列为24/48/168小时；Data列为25/50/100\%；表内为季节性MASE。\normalsize
'''+figure('sensitivity','单种子受控敏感性；连接线仅连接已测配置，不表示插值性能或显著性。','5.7cm')+r'''
\subsection{可以支持的解释与混杂因素}
'''+sensitivity_statement('Electricity','window','PatchTST',True)+sensitivity_statement('Solar','window','LSTM',True)+sensitivity_statement('Electricity','fraction','LSTM',True)+r'''
端点差异不证明单调收益，仍需阅读中间配置。增加历史改变patch数量和输出头容量；即便循环模型参数量不变，计算量也改变。这是历史与容量共同变化的敏感性，不能作为纯记忆机制贡献证明。

缩减连续训练片段同时改变季节覆盖、数据新旧构成，以及五轮内的优化步数。各比例重新估计MASE分母，因此跨比例MASE变化还可能包含参照尺度变化；需结合JSON中的原尺度MAE/RMSE，不把所有差异归因于样本数量。多种子敏感性与固定更新步数控制留到终稿。
''')
    p.append(section(9,'工程判断、局限与终稿计划',True)+r'''
\subsection{暂定工程选择}
应按任务比较精度与成本，而不是追求单个架构冠军。Electricity需要把patch模型的改善，与零参数日周期基线和轻量DLinear对照；Solar在当前信息与预算下，日周期基线是较强参照和可行的初始回退方案。复杂模型应通过高负荷或活跃出力等应用相关评价证明额外成本值得承担，当前误差变化没有直接转化为经济收益或控制效果。

参数量描述权重存储，峰值内存还包含激活与工作空间；两者都不能单独证明嵌入式可部署性。算子支持、数值精度、运行时与端到端延迟必须在目标设备检查。本地GPU测量不能推断T4吞吐或微控制器效果。任何帕累托判断都只对选定目标与测量条件有效，不是普遍最优。
\subsection{影响解释的局限}
\begin{enumerate}
\item 范围：固定16对象、单个时间区间不能证明其他用户、电站、年份或地区上的泛化。Solar有模拟来源，且天文和时区语义未独立验证。
\item 训练：五轮和通用设置属于初步预算，不意味着各架构已公平充分优化。三个种子只描述部分训练随机性，部分验证曲线到预算上限仍在改善。
\item 机制：容量、物理滤波跨度、归一化和输入表示都可能混杂架构比较；自写PatchTST风格并非完整原模型。当前不研究预训练基础模型、外生变量或跨数据集迁移。
\item 评价：窗口重叠使误差相关；MASE、原尺度指标和条件切片的权重不同。历史迭代曾查看测试区间，不能称为全新外部盲测。新版设置预先固定，未按测试排名调参或选种子。
\end{enumerate}
\subsection{终稿的具体完成路径}
\begin{enumerate}
\item 验证与预算：检查曲线，增加到最多20轮，并进行规模受限、可记录的验证集搜索；记录全部尝试，控制不同模型调参预算，不挑最好测试种子。
\item 敏感性与消融：各设置补多种子，同时报告MAE/RMSE与MASE；补固定更新步数控制、容量匹配的CNN前端比较，继续冻结通道与共同起点。
\item 误差分析：补逐通道结果；确认时间语义后再做工作日/周末；增加预定困难场景、残差依赖和统一截断比较。若需统计推断，应按时间块处理依赖，不能把窗口当独立样本。
\item 交付：确认姓名、学号与各节真实作者；终稿按另外的两人总计20页限制整理，并保留代码、数据库证据，准备课程口试和最多3分钟视频\cite{course}。
\end{enumerate}
\subsection{初步结论}
本次核验说明，基线强度和实验协议应先于模型复杂度。Electricity上的patch编码值得继续验证，Solar则暴露出当前预算下与日周期基线的明显差距。现阶段成果是可复现的初步比较、与实际实现一致的机制解释，以及检验这些观察能否在更充分训练和更广验证下成立的受控计划。
''')
    return p

if __name__=='__main__':
    print(build(False));print(build(True))
