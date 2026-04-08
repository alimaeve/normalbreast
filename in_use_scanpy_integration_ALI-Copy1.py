#!/usr/bin/env python
# coding: utf-8

# # Loading Data #

# In[2]:


import scanpy as sc
import pandas as pd
import anndata
import numpy as np
import scvi
from scipy.sparse import csr_matrix
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster #, sch
from scipy.spatial.distance import squareform, pdist
sc.settings.verbosity = 0             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_versions()
sc.settings.set_figure_params(dpi=80, frameon=False, figsize=(3, 3), facecolor='white')
sc.settings.seed = 1
import warnings
warnings.filterwarnings("ignore")
from matplotlib.colors import LinearSegmentedColormap
from gseapy import enrichr, barplot, dotplot


# In[2]:


ind1_KN23_data = pd.read_csv("./data/Kevin_Nee_2021/GSM5320152_scRNA_ctrl1_matrix.txt.gz", delim_whitespace=True).T
ind2_KN23_data = pd.read_csv("./data/Kevin_Nee_2021/GSM5320164_scRNA_brca2_matrix.txt.gz", delim_whitespace=True).T
ind3_KN23_data = pd.read_csv("./data/Kevin_Nee_2021/GSM5320165_scRNA_brca3_matrix.txt.gz", delim_whitespace=True).T
ind4_KN23_data = pd.read_csv("./data/Kevin_Nee_2021/GSM5320172_scRNA_brca10_matrix.txt.gz", delim_whitespace=True).T
ind5_KN23_data = pd.read_csv("./data/Kevin_Nee_2021/GSM5320163_scRNA_brca1_matrix.txt.gz", delim_whitespace=True).T
ind6_KN23_data = pd.read_csv("./data/Kevin_Nee_2021/GSM5320166_scRNA_brca4_matrix.txt.gz", delim_whitespace=True).T
ind7_KN23_data = pd.read_csv("./data/Kevin_Nee_2021/GSM5320168_scRNA_brca6_matrix.txt.gz", delim_whitespace=True).T
ind8_KN23_data = pd.read_csv("./data/Kevin_Nee_2021/GSM5320173_scRNA_brca11_matrix.txt.gz", delim_whitespace=True).T
ind9_KN23_data = pd.read_csv("./data/Kevin_Nee_2021/GSM5320153_scRNA_ctrl2_matrix.txt.gz", delim_whitespace=True).T
ind10_KN23_data = pd.read_csv("./data/Kevin_Nee_2021/GSM5320154_scRNA_ctrl3_matrix.txt.gz", delim_whitespace=True).T
ind11_KN23_data = pd.read_csv("./data/Kevin_Nee_2021/GSM5320157_scRNA_ctrl6_matrix.txt.gz", delim_whitespace=True).T
ind12_KN23_data = pd.read_csv("./data/Kevin_Nee_2021/GSM5320171_scRNA_brca9_matrix.txt.gz", delim_whitespace=True).T
ind13_KN23_data = pd.read_csv("./data/Kevin_Nee_2021/GSM5320169_scRNA_brca7_matrix.txt.gz", delim_whitespace=True).T


# In[ ]:


ind01_KN23_data = sc.AnnData(X=ind1_KN23_data.values)
ind01_KN23_data.obs_names = ind1_KN23_data.index  
ind01_KN23_data.var_names = ind1_KN23_data.columns  

ind02_KN23_data = sc.AnnData(X=ind2_KN23_data.values)
ind02_KN23_data.obs_names = ind2_KN23_data.index  
ind02_KN23_data.var_names = ind2_KN23_data.columns 

ind03_KN23_data = sc.AnnData(X=ind3_KN23_data.values)
ind03_KN23_data.obs_names = ind3_KN23_data.index  
ind03_KN23_data.var_names = ind3_KN23_data.columns 

ind04_KN23_data = sc.AnnData(X=ind4_KN23_data.values)
ind04_KN23_data.obs_names = ind4_KN23_data.index  
ind04_KN23_data.var_names = ind4_KN23_data.columns 

ind05_KN23_data = sc.AnnData(X=ind5_KN23_data.values)
ind05_KN23_data.obs_names = ind5_KN23_data.index  
ind05_KN23_data.var_names = ind5_KN23_data.columns 

ind06_KN23_data = sc.AnnData(X=ind6_KN23_data.values)
ind06_KN23_data.obs_names = ind6_KN23_data.index  
ind06_KN23_data.var_names = ind6_KN23_data.columns 

ind07_KN23_data = sc.AnnData(X=ind7_KN23_data.values)
ind07_KN23_data.obs_names = ind7_KN23_data.index  
ind07_KN23_data.var_names = ind7_KN23_data.columns 

ind08_KN23_data = sc.AnnData(X=ind8_KN23_data.values)
ind08_KN23_data.obs_names = ind8_KN23_data.index  
ind08_KN23_data.var_names = ind8_KN23_data.columns 

ind09_KN23_data = sc.AnnData(X=ind9_KN23_data.values)
ind09_KN23_data.obs_names = ind9_KN23_data.index  
ind09_KN23_data.var_names = ind9_KN23_data.columns 

ind010_KN23_data = sc.AnnData(X=ind10_KN23_data.values)
ind010_KN23_data.obs_names = ind10_KN23_data.index  
ind010_KN23_data.var_names = ind10_KN23_data.columns 

ind011_KN23_data = sc.AnnData(X=ind11_KN23_data.values)
ind011_KN23_data.obs_names = ind11_KN23_data.index  
ind011_KN23_data.var_names = ind11_KN23_data.columns 

ind012_KN23_data = sc.AnnData(X=ind12_KN23_data.values)
ind012_KN23_data.obs_names = ind12_KN23_data.index  
ind012_KN23_data.var_names = ind12_KN23_data.columns 

ind013_KN23_data = sc.AnnData(X=ind13_KN23_data.values)
ind013_KN23_data.obs_names = ind13_KN23_data.index  
ind013_KN23_data.var_names = ind13_KN23_data.columns 


# In[4]:


#ind1_KK18_data = sc.read_csv("./data/Kai_Kessenbrock_2018/GSE113099_individual.1.grch.fpkm.matrix.no.doublets.or.empty.txt.gz", delimiter='\t').T
#ind2_KK18_data = sc.read_csv("./data/Kai_Kessenbrock_2018/GSE113127_individual.2.grch.fpkm.matrix.no.doublets.or.empty.txt.gz", delimiter='\t').T
#ind3_KK18_data = sc.read_csv("./data/Kai_Kessenbrock_2018/GSE113198_individual.3.grch.fpkm.matrix.no.doublets.or.empty.txt.gz", delimiter='\t').T
ind4_KK18_data = sc.read_csv("./data/Kai_Kessenbrock_2018/GSM3099846_Ind4_Expression_Matrix.txt.gz", delimiter='\t').T
ind5_KK18_data = sc.read_csv("./data/Kai_Kessenbrock_2018/GSM3099847_Ind5_Expression_Matrix.txt.gz", delimiter='\t').T
ind6_KK18_data = sc.read_csv("./data/Kai_Kessenbrock_2018/GSM3099848_Ind6_Expression_Matrix.txt.gz", delimiter='\t').T
ind7_KK18_data = sc.read_csv("./data/Kai_Kessenbrock_2018/GSM3099849_Ind7_Expression_Matrix.txt.gz", delimiter='\t').T


# In[5]:


ind1_N2021_data = sc.read_10x_h5("./data/Nakshatri_2021/GSM5022599_D1_filtered_feature_bc_matrix.h5")
ind2_N2021_data = sc.read_10x_h5("./data/Nakshatri_2021/GSM5022600_D2_filtered_feature_bc_matrix.h5")
ind3_N2021_data = sc.read_10x_h5("./data/Nakshatri_2021/GSM5022601_D3_filtered_feature_bc_matrix.h5")
ind4_N2021_data = sc.read_10x_h5("./data/Nakshatri_2021/GSM5022602_D4_filtered_feature_bc_matrix.h5")
ind5_N2021_data = sc.read_10x_h5("./data/Nakshatri_2021/GSM5022603_D5_filtered_feature_bc_matrix.h5")

ind1_N2021_data.var_names_make_unique()
ind2_N2021_data.var_names_make_unique()
ind3_N2021_data.var_names_make_unique()
ind4_N2021_data.var_names_make_unique()
ind5_N2021_data.var_names_make_unique()


# In[6]:


ind1_JV2021_data = sc.read_10x_mtx("./data/Jane_Visvader_2021/", prefix="GSM4909253_N-PM0092-Total-")
ind2_JV2021_data = sc.read_10x_mtx("./data/Jane_Visvader_2021/", prefix="GSM4909254_N-PM0019-Total-")
ind3_JV2021_data = sc.read_10x_mtx("./data/Jane_Visvader_2021/", prefix="GSM4909255_N-N280-Epi-")
ind4_JV2021_data = sc.read_10x_mtx("./data/Jane_Visvader_2021/", prefix="GSM4909256_N-PM0095-Epi-")
ind5_JV2021_data = sc.read_10x_mtx("./data/Jane_Visvader_2021/", prefix="GSM4909258_N-NF-Epi-")
ind6_JV2021_data = sc.read_10x_mtx("./data/Jane_Visvader_2021/", prefix="GSM4909259_N-NE-Epi-")
ind7_JV2021_data = sc.read_10x_mtx("./data/Jane_Visvader_2021/", prefix="GSM4909260_N-N1105-Epi-")
ind8_JV2021_data = sc.read_10x_mtx("./data/Jane_Visvader_2021/", prefix="GSM4909261_N-PM0230-Total-")
ind9_JV2021_data = sc.read_10x_mtx("./data/Jane_Visvader_2021/", prefix="GSM4909262_N-MH0064-Epi-")
ind10_JV2021_data = sc.read_10x_mtx("./data/Jane_Visvader_2021/", prefix="GSM4909264_N-N1B-Epi-")


# In[7]:


ind11_JV2021_data = sc.read_10x_mtx("./data/Jane_Visvader_2021/", prefix="GSM4909265_N-PM0233-Total-")
ind12_JV2021_data = sc.read_10x_mtx("./data/Jane_Visvader_2021/", prefix="GSM4909266_N-MH0169-Total-")
ind13_JV2021_data = sc.read_10x_mtx("./data/Jane_Visvader_2021/", prefix="GSM4909267_N-MH0023-Epi-")
ind14_JV2021_data = sc.read_10x_mtx("./data/Jane_Visvader_2021/", prefix="GSM4909269_N-PM0342-Epi-")
ind15_JV2021_data = sc.read_10x_mtx("./data/Jane_Visvader_2021/", prefix="GSM4909271_N-MH288-Total-")
ind16_JV2021_data = sc.read_10x_mtx("./data/Jane_Visvader_2021/", prefix="GSM4909272_N-MH0021-Total-")
ind17_JV2021_data = sc.read_10x_mtx("./data/Jane_Visvader_2021/", prefix="GSM4909273_N-MH275-Epi-")
ind18_JV2021_data = sc.read_10x_mtx("./data/Jane_Visvader_2021/", prefix="GSM4909275_N-PM0372-Epi-")


# In[8]:


ind1_T2022_data = sc.read_10x_mtx("./data/Tokura_2022/", prefix="GSM5852274_NCCBC13_filtered_feature_bc_matrix_")


# In[9]:


ind1_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20005-20446_SIGAG11_")
ind2_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-19864-20262_SIGAH5_")
ind3_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20005-20446_SIGAD12_")
ind4_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20870_SIGAC9_")
ind5_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20261-20448_SIGAC6_")
ind6_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20870_SIGAB9_")
ind7_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20870_SIGAA9_")
ind8_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20870_SIGAG9_")
ind9_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20261-20448_SIGAE6_")


# In[10]:


ind10_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20005_SIGAH11_")
ind11_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20261-20448_SIGAH6_")
ind12_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20005-20446_SIGAC6_")
ind13_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20005-20446_SIGAE11_")
ind14_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20261_SIGAF6_")
ind15_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20870_SIGAD9_")
ind16_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20005-20446_SIGAB6_")
ind17_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-19864_SIGAD8_")
ind18_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20261-20448_SIGAG6_")
ind19_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20005-20446_SIGAA7_")


# In[11]:


ind20_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20261-20448_SIGAB1_")
ind21_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20261_SIGAA6_")
ind22_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-19864-20262_SIGAC8_")
ind23_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-19902-20449_SIGAA3_")
ind24_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-19902-20449_SIGAF1_")
ind25_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-19902-20449_SIGAD1_")
ind26_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-19864-20262_SIGAE6_")
ind27_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-19864-20262_SIGAE5_")
ind28_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-19864-20262_SIGAG6_")
ind29_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20005-20446_SIGAD7_")


# In[12]:


ind30_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20005-20446_SIGAC12_")
ind31_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20005-20446_SIGAC7_")
ind32_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-19864-20262_SIGAG5_")
ind33_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20261-20448_SIGAC1_")
ind34_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-19864-20262_SIGAD10_")
ind35_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20005-20446_SIGAB12_")
ind36_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-19864-20262_SIGAF8_")
ind37_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-19864-20262_SIGAE8_")
ind38_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-19647-20870_SIGAB8_")
ind39_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-19902-20449_SIGAB3_")


# In[13]:


ind40_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20005_SIGAA12_")
ind41_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20261-20448_SIGAA1_")
ind42_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20261-20448_SIGAB6_")
ind43_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-19864-20262_SIGAF6_")
ind44_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-19864-20262_SIGAE10_")
ind45_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20261-20448_SIGAD6_")
ind46_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20005-20446_SIGAD6_")
ind47_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20261-20448_SIGAH9_")
ind48_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-19902-20449_SIGAG1_")
ind49_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-19864_SIGAH6_")


# In[14]:


ind50_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20005_SIGAB7_")
ind51_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-19902-20449_SIGAD3_")
ind52_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-19902-20449_SIGAC3_")
ind53_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-19902-20449_SIGAE1_")
ind54_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20005-20446_SIGAF11_")
ind55_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-20261-20448_SIGAG10_")
ind56_R2024_data = sc.read_10x_mtx("./data/Reed_2024/", prefix="SLX-19864_SIGAF5_")


# In[15]:


inds_JB2022_data = sc.read_csv("./data/Joan_Brugge_2022/GSE180878_Li_Brugge_10XscRNAseq_GeneCellMatrix_RNAcounts_human.csv.gz")
inds_JB2022_data = inds_JB2022_data.T
inds_JB2022_data = inds_JB2022_data[list(map(lambda x: x.startswith("RM-"), inds_JB2022_data.obs.index.values)), :]
inds_JB2022_data.obs["Sample"] = list(map(lambda x: x.split("_")[0], inds_JB2022_data.obs.index.values))


# In[16]:


inds_NN2023_data = sc.read_h5ad("./data/Navin_2023/e82304a5-8a35-4de9-9659-812cda712665.h5ad")
#use original authors pre-filtering
inds_NN2023_data.obs['Sample'] = 'NN2023'
inds_NN2023_data.obs['Batch'] = 'NN2023'
inds_NN2023_data.obs['Experiment'] = '10x Genomics'

#rename columns for scanpy compatibility later
inds_NN2023_data.obs['n_genes_by_counts'] = inds_NN2023_data.obs['n_feature_rna']
inds_NN2023_data.obs['pct_counts_mt'] = inds_NN2023_data.obs['percent_mito']
inds_NN2023_data.obs['pct_counts_ribo'] = inds_NN2023_data.obs['percent_rb']


# # Preprocessing

# In[ ]:


ribo_url = "http://software.broadinstitute.org/gsea/msigdb/download_geneset.jsp?geneSetName=KEGG_RIBOSOME&fileType=txt"
ribo_genes = pd.read_table(ribo_url, skiprows=2, header = None)


# In[ ]:


def pp(adata, samplename, batch, expdesign=None, thr_mito=20, thr_ribo=2, verbose=True):
    if not 'Sample' in adata.obs:
        adata.obs['Sample'] = samplename
    else:
        print("sample is already there, which might be ok")
    adata.obs['Batch'] = batch
    adata.obs['Experiment'] = expdesign
    sc.pp.filter_cells(adata, min_genes=200) #get rid of cells with fewer than 200 genes
    sc.pp.filter_genes(adata, min_cells=3) #get rid of genes that are found in fewer than 3 cells
    if verbose:
        print(adata.shape)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    adata.var['ribo'] = adata.var_names.isin(ribo_genes[0].values)
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo'], percent_top=None, log1p=False, inplace=True)
    if verbose:
        sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
        sc.pl.scatter(adata, x='total_counts', y='pct_counts_ribo')
    upper_lim = np.quantile(adata.obs.n_genes_by_counts.values, .98)
    adata = adata[adata.obs.n_genes_by_counts < upper_lim]
    adata = adata[adata.obs.pct_counts_mt < thr_mito]
    if verbose:
        print(adata.shape)
    adata = adata[adata.obs.pct_counts_ribo > thr_ribo]
    if verbose:
        print(adata.shape)
    return adata


# In[ ]:


# Do for KN23

designs = ["10x Genomics"]*13
samples = [f"ind{i}_KN23" for i in range(1,14)]

dataobjects_KN23 = ["ind01_KN23_data", "ind02_KN23_data", "ind03_KN23_data", "ind04_KN23_data", "ind05_KN23_data", "ind06_KN23_data", 
                   "ind07_KN23_data", "ind08_KN23_data", "ind09_KN23_data", "ind010_KN23_data", "ind011_KN23_data", "ind012_KN23_data",
                   "ind013_KN23_data"]

for i,d in enumerate(dataobjects_KN23):
    exec(f"{dataobjects_KN23[i]} = pp({d}, samples[i], 'KN23', expdesign=designs[i], thr_mito=20, thr_ribo=5, verbose=False)")


# In[20]:


# Do for KK18

designs = ["10x Genomics"]*4
samples = [f"ind{i}_KK18" for i in range(4,8)]
#dataobjects_KK18 = ["ind1_KK18_data", "ind2_KK18_data", "ind2_KK18_data"]

dataobjects_KK18 = ["ind4_KK18_data", "ind5_KK18_data", "ind6_KK18_data", "ind7_KK18_data"]

for i,d in enumerate(dataobjects_KK18):
    exec(f"{dataobjects_KK18[i]} = pp({d}, samples[i], 'KK18', expdesign=designs[i], thr_mito=20, thr_ribo=5, verbose=False)")


# In[21]:


#Do for N2021

designs = ["10x Genomics"]*5
samples = [f"ind{i}_N2021" for i in range(1,6)]
dataobjects_N2021 = ["ind1_N2021_data", "ind2_N2021_data", "ind3_N2021_data",
                    "ind4_N2021_data", "ind5_N2021_data"]

for i,d in enumerate(dataobjects_N2021):
    exec(f"{dataobjects_N2021[i]} = pp({d}, samples[i], 'N2021', expdesign=designs[i], thr_mito=20, thr_ribo=5, verbose=False)")


# In[22]:


# Do for JV2021

designs = ["TruSeq"]*18
samples = [f"ind{i}_JV2021" for i in range(1,19)]
dataobjects_JV2021 = ["ind1_JV2021_data", "ind2_JV2021_data", "ind3_JV2021_data",
                    "ind4_JV2021_data", "ind5_JV2021_data", "ind6_JV2021_data",
                    "ind7_JV2021_data", "ind8_JV2021_data", "ind9_JV2021_data",
                    "ind10_JV2021_data", "ind11_JV2021_data", "ind12_JV2021_data",
                     "ind13_JV2021_data", "ind14_JV2021_data", "ind15_JV2021_data",
                     "ind16_JV2021_data", "ind17_JV2021_data", "ind18_JV2021_data"]

for i,d in enumerate(dataobjects_JV2021):
    exec(f"{dataobjects_JV2021[i]} = pp({d}, samples[i], 'JV2021', expdesign=designs[i], thr_mito=20, thr_ribo=5, verbose=False)")


# In[23]:


# Do for T2020

designs = ["10x Genomics"]*1
samples = [f"ind{i}_T2020" for i in range(1)]
dataobjects_T2020 = ["ind1_T2022_data"]
for i,d in enumerate(dataobjects_T2020):
   exec(f"{dataobjects_T2020[i]} = pp({d}, samples[i], 'T2020', expdesign=designs[i], thr_mito=20, thr_ribo=5, verbose=False)")


# In[24]:


# Do for JB2022

inds_JB2022_data = pp(inds_JB2022_data, None, 'JB2022', expdesign="10x Genomics", thr_mito=20, thr_ribo=5, verbose=True)


# In[25]:


# Do for R2024

designs = ["10x Genomics"]*56
samples = [f"ind{i}_R2024" for i in range(1,57)]
dataobjects_R2024 = ["ind1_R2024_data", "ind2_R2024_data", "ind3_R2024_data",
                    "ind4_R2024_data", "ind5_R2024_data", "ind6_R2024_data", "ind7_R2024_data", "ind8_R2024_data", "ind9_R2024_data",
                    "ind10_R2024_data", "ind11_R2024_data", "ind12_R2024_data","ind13_R2024_data", "ind14_R2024_data", "ind15_R2024_data",
                    "ind16_R2024_data", "ind17_R2024_data", "ind18_R2024_data","ind19_R2024_data", "ind20_R2024_data", "ind21_R2024_data",
                    "ind22_R2024_data", "ind23_R2024_data", "ind24_R2024_data","ind25_R2024_data", "ind26_R2024_data", "ind27_R2024_data",
                    "ind28_R2024_data", "ind29_R2024_data", "ind30_R2024_data","ind31_R2024_data", "ind32_R2024_data", "ind33_R2024_data",
                    "ind34_R2024_data", "ind35_R2024_data", "ind36_R2024_data","ind37_R2024_data", "ind38_R2024_data", "ind39_R2024_data",
                    "ind40_R2024_data", "ind41_R2024_data", "ind42_R2024_data","ind43_R2024_data", "ind44_R2024_data", "ind45_R2024_data",
                    "ind46_R2024_data", "ind47_R2024_data", "ind48_R2024_data","ind49_R2024_data", "ind50_R2024_data", "ind51_R2024_data",
                    "ind52_R2024_data", "ind53_R2024_data", "ind54_R2024_data","ind55_R2024_data", "ind56_R2024_data"]
for i,d in enumerate(dataobjects_R2024):
    exec(f"{dataobjects_R2024[i]} = pp({d}, samples[i], 'R2024', expdesign=designs[i], thr_mito=20, thr_ribo=5, verbose=False)")


# In[26]:


BRCA1 = ["RM-D","ind13_JV2021", "ind1_R2024","ind3_R2024","ind4_R2024","ind6_R2024","ind7_R2024", "ind10_R2024","ind13_R2024","ind15_R2024",
        "ind19_R2024","ind39_R2024","ind32_R2024","ind22_R2024","ind23_R2024","ind25_R2024","ind27_R2024","ind30_R2024"]
BRCA2 = ["ind37_R2024","ind47_R2024","ind52_R2024","ind46_R2024","ind40_R2024","ind36_R2024","ind5_R2024","ind34_R2024","ind31_R2024","ind18_R2024","ind21_R2024","ind24_R2024","ind28_R2024"]
WT = ["ind9_R2024", "ind16_R2024", "RM-A", "RM-B", "RM-C"]
Unknown = ["ind1_T2022","ind4_KK18", "ind5_KK18", "ind6_KK18", "ind7_KK18", "ind1_N2021", "ind2_N2021", "ind3_N2021",
                    "ind4_N2021", "ind5_N2021", "ind1_JV2021", "ind2_JV2021", "ind3_JV2021",
                    "ind4_JV2021","ind45_R2024","ind44_R2024","ind43_R2024","ind42_R2024", "ind5_JV2021", "ind6_JV2021",
                    "ind7_JV2021","ind41_R2024", "ind8_JV2021", "ind9_JV2021",
                    "ind10_JV2021","ind51_R2024","ind56_R2024","ind55_R2024","ind54_R2024","ind53_R2024","ind50_R2024","ind49_R2024","ind48_R2024", "ind11_JV2021", "ind12_JV2021",
                     "ind14_JV2021", "ind15_JV2021",
                     "ind16_JV2021", "ind17_JV2021", "ind18_JV2021", "ind2_R2024","ind8_R2024","ind11_R2024","ind12_R2024","ind14_R2024","ind17_R2024",
          "ind20_R2024","ind38_R2024","ind35_R2024","ind33_R2024","ind26_R2024","ind29_R2024"]
def get_mutation_status(sample):
    if sample in BRCA1:
        return "BRCA1"
    elif sample in BRCA2:
        return "BRCA2"
    elif sample in WT:
        return "WT"
    elif sample in Unknown:
        return "Unknown"
    else:
        return "Not Found"


# In[26]:


#change ensembl id's as var in NN2023 to the symbols stored
inds_NN2023_data.var_names = inds_NN2023_data.var['feature_name'].astype(str)
inds_NN2023_data.var_names_make_unique()
inds_NN2023_data.var_names.str.startswith('ENSG').sum()
#remove genes with ENSG names, that haven't been mapped to gene symbols
symbol_mask = ~inds_NN2023_data.var_names.str.startswith('ENSG')
inds_NN2023_data = inds_NN2023_data[:, symbol_mask]


# In[27]:


ind01_KN23_data.var_names


# # Concatenating Samples & Analysed

# In[28]:


alldatasets = dataobjects_N2021 + dataobjects_R2024 + dataobjects_KK18 + dataobjects_N2021 + dataobjects_JV2021 + dataobjects_KN23 + ["inds_JB2022_data"] + ["inds_NN2023_data"]

alldata_KK18 = sc.concat([globals()[d] for d in alldatasets])


# In[30]:


alldata_KK18.write_h5ad('260126_normal_combined_after_pp.h5ad')


# In[3]:


alldata_KK18 = sc.read_h5ad('260126_normal_combined_after_pp.h5ad')


# In[13]:


'NEP' in alldata_KK18.var_names


# In[11]:


alldata_KK18.var_names[:20]


# In[8]:


print(alldata_KK18.shape)
sc.pp.filter_genes(alldata_KK18, min_cells = 200)
alldata_KK18.X = csr_matrix(alldata_KK18.X)
print(alldata_KK18.shape)


# In[ ]:


print(alldata_KK18)


# In[ ]:


alldata_KK18.obs.groupby('Sample').count()


# In[ ]:


alldata_KK18.layers['counts'] = alldata_KK18.X.copy()
sc.pp.normalize_total(alldata_KK18, target_sum = 1e4)
sc.pp.log1p(alldata_KK18)
alldata_KK18.raw = alldata_KK18
alldata_KK18.obs.head()


# In[ ]:


sc.pp.highly_variable_genes(alldata_KK18, n_top_genes = 2000)
alldata_KK18_hv = alldata_KK18[:, alldata_KK18.var['highly_variable']]
sc.pp.pca(alldata_KK18_hv)
sc.pp.neighbors(alldata_KK18_hv, n_pcs = 20)
sc.tl.leiden(alldata_KK18_hv, resolution = 0.8)
sc.tl.umap(alldata_KK18_hv)


# In[ ]:


alldata_KK18.obs["leiden"] = alldata_KK18_hv.obs["leiden"]


# In[ ]:


sc.pl.umap(alldata_KK18_hv, color=['Batch'])


# In[ ]:


sc.pl.umap(alldata_KK18_hv, color=['leiden', 'Batch', 'Sample'], ncols=1)


# In[24]:


alldata_KK18_hv.layers["counts"].toarray()


# In[15]:


sc.pl.umap(alldata_KK18_hv, color = ['leiden'], frameon = False, legend_loc = "on data")


# In[16]:


sc.pl.umap(alldata_KK18_hv, color=['Sample'])


# In[4]:


alldata_KK18_hv.write_h5ad('260126_normal_alldata_KK18_hv.h5ad')


# In[5]:


alldata_KK18_hv = sc.read_h5ad('260126_normal_alldata_KK18_hv.h5ad')


# In[7]:


#francesca's marker gene 
sc.pl.umap(alldata_KK18_hv, color=['MME'], ncols=1)


# In[27]:


#BASAL Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(alldata_KK18_hv, color=['ACTA2', 'TAGLN', 'MYL9', 'TPM2', 'ACTG2', 'KRT5', 'ITGA6', 'KRT14', 'KRT17', 'CCND2', 'DKK3', 'SPARC', 'KRT14', 'NFIB','ACTG2','MYLK','SAA1','DST','LAMB3'], ncols=4)


# In[28]:


#LUMINAL PROGENITOR Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(alldata_KK18_hv, color=['LTF', 'SLPI', 'RARRES1', 'ALDH1A3', 'KRT15', 'PI3', 'S100A9', 'CD24', 'KIT', 'SFRP1','LTF'], ncols=4)


# In[29]:


#MATURE LUMINAL Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(alldata_KK18_hv, color=['AREG', 'PRLR', 'PGR', 'ESR1', 'FOXA1', 'TTF1', 'TFF3', 'SYTL2', 'AKT1', 'IGF1R', 'ALCAM','STC2', 'GATA3', 'XBP1'], ncols=4)


# In[30]:


#EPCAM & KRT19 & ITGA6 (CD49F)#
sc.pl.umap(alldata_KK18_hv, color=['KRT19', 'ITGA6'], ncols=3)


# In[31]:


sc.tl.rank_genes_groups(alldata_KK18, 'leiden')
markers = sc.get.rank_genes_groups_df(alldata_KK18, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > .5)]
markers


# In[39]:


marker_genes = {
   # "Epithelial": ['EPCAM'],
    "Luminal":['KRT19'],
    "Basal": ['TAGLN', 'KRT14', 'ACTA2', 'KRT17', 'SAA1', 'MYLK', 'KRT5'], #TP63
    "Luminal_Mature": ['FOXA1', 'ESR1', 'AREG'], #MUCL1, PIP, AGR2
    "Luminal_Progenitor": ['KRT15', 'LTF', 'SLPI'], #ELF5, EHF, PROM1, GABRP
    "Adipocyte": ['APOE'], #ADIPOQ
    "Endothelial": ['FABP5', 'MECOM'], #PECAM1, CLDN5, CDH5, FABP4
    "Fibroblast": ['APOD', 'COL1A1', 'TNFAIP6'], #DCN, COL1A2
    "General_Myeloid": ['HLA-DRA', 'HLA-DPA1', 'CD74'],
    "Monocyte": ['VCAN', 'CD14'],
    "Macrophage": ['APOE'], #CCL3, CCL4, IL1B, IDO1
    "T-Cell": ['CCL5', 'CXCR4'] #CD2, GNLY, IL7R, PTPRC
  #  "B-Cell": [ ] #CD79B, IGKC
}

sc.pl.dotplot(
    alldata_KK18,
    var_names=marker_genes,
    groupby='leiden'
)


# In[32]:


sc.pl.rank_genes_groups(alldata_KK18, n_genes=20, sharey=False)


# ### Cell type markers by clusters

# In[46]:


cell_type = {"0":"Endothelial",
"1":"Fibroblast",
"2":"Basal",
"3":"Endothelial",
"4":"Fibroblast",
"5":"Endothelial",
"6":"Luminal Mature",
"7":"Luminal Progenitor",
"8":"Luminal Progenitor",
"9":"Basal",
"10":"Endothelial",
"11":"Endothelial",
"12":"Luminal Mature",
"13":"Luminal Mature",
"14":"Fibroblast",
"15":"Luminal Mature",
"16":"T-Cell",
"17":"Endothelial",
"18":"Luminal Progenitor",
"19":"Basal",
"20":"Luminal Mature",
"21":"Luminal Progenitor",
"22":"Basal",
"23":"Luminal Progenitor",
"24":"Fibroblast",
"25":"General Myeloid",
"26":"Luminal Progenitor",
"27":"Endothelial", "28":"General Myeloid", "29":"Luminal Progenitor", "30":"Fibroblast", "31":"General Myeloid", "32":"Luminal Progenitor"}


# In[47]:


alldata_KK18_hv.obs['cell type'] = alldata_KK18_hv.obs.leiden.map(cell_type)


# In[48]:


sc.pl.umap(alldata_KK18_hv, color = ['cell type'], frameon = False)


# In[55]:


filtered_out_immune = alldata_KK18_hv[alldata_KK18_hv.obs['leiden'].isin(['2','6','7','8','9','12','13','15','18', '19','20','21','22', '23', '26', '29', '32'])]


# In[56]:


sc.pl.umap(filtered_out_immune, color = ['cell type'], frameon = False)


# In[57]:


sc.pl.umap(filtered_out_immune, color = ['leiden'], frameon = False, legend_loc = "on data")


# In[52]:


#save annotated anndata object for later:
alldata_KK18_hv.write_h5ad('260126_normal_alldata_hv.h5ad') # save for later
#alldata_KK18_hv.write_h5ad("./data/28_06_2025_alldata_KK18_not_hv_not_annotated.h5ad")


# In[54]:


#Francesca CD90 aka THY1Basal Cells?#
sc.pl.umap(alldata_KK18_hv, color=['THY1'], ncols=1)


# In[60]:


cell_types_of_interest = ['Basal', 'Luminal Mature', 'Luminal Progenitor']
if 'Batch' in alldata_KK18_hv.obs.columns:
    counts_per_study = alldata_KK18_hv.obs.groupby('Batch')['cell type'].value_counts().unstack(fill_value=0)
    counts_per_study_of_interest = counts_per_study[cell_types_of_interest]
    perc_per_study = counts_per_study_of_interest.div(counts_per_study_of_interest.sum(axis=1), axis=0) * 100
    counts_per_study_of_interest.to_csv("cell_counts_per_study.csv")
    perc_per_study.to_csv("cell_percentages_per_study.csv")


# In[61]:


epithelial_cell_types = ['Luminal Mature', 'Luminal Progenitor', 'Basal']
subset = alldata_KK18_hv[alldata_KK18_hv.obs['cell type'].isin(epithelial_cell_types)].copy()
subset.write('20260126_normal_luminal_basal_subset.h5ad')


# In[3]:


#reload just basal, luminal mature & luminal progenitor cells
basal_luminal_cells = sc.read_h5ad("./luminal_basal_subset.h5ad")
print(basal_luminal_cells)
print(basal_luminal_cells.obs['cell type'].head())


# In[10]:


sc.pl.umap(basal_luminal_cells, color=['cell type'])


# # KN23 Analysed Alone

# In[ ]:


KN23_only = sc.concat([globals()[d] for d in dataobjects_KN23])


# In[ ]:


print(KN23_only.shape)
sc.pp.filter_genes(KN23_only, min_cells = 200)
KN23_only.X = csr_matrix(KN23_only.X)
print(KN23_only.shape)


# In[ ]:


KN23_only.obs.groupby('Sample').count()


# In[ ]:


KN23_only.layers['counts'] = KN23_only.X.copy()
sc.pp.normalize_total(KN23_only, target_sum = 1e4)
sc.pp.log1p(KN23_only)
KN23_only.raw = KN23_only
KN23_only.obs.head()


# In[ ]:


sc.pp.highly_variable_genes(KN23_only, n_top_genes = 2000)
KN23_only_hv = KN23_only[:, KN23_only.var['highly_variable']].copy()
sc.pp.pca(KN23_only_hv, n_comps=50)
sc.pp.neighbors(KN23_only_hv, n_pcs = 20)


# In[ ]:


sc.tl.leiden(KN23_only_hv, resolution = 0.4)
sc.tl.umap(KN23_only_hv)
sc.pl.umap(KN23_only_hv, color=['leiden', 'Batch', 'Sample'])


# In[ ]:


#BASAL Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(KN23_only_hv, color=['ACTA2', 'TAGLN', 'MYL9', 'TPM2', 'ACTG2', 'KRT5', 'ITGA6', 'KRT14', 'KRT17', 'CCND2', 'DKK3', 'SPARC'], ncols=4)


# In[ ]:


#LUMINAL PROGENITOR Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(KN23_only_hv, color=['LTF', 'SLPI', 'RARRES1', 'ALDH1A3', 'KRT15', 'PI3', 'S100A9', 'CD24', 'KIT'], ncols=4)


# In[ ]:


#MATURE LUMINAL Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(KN23_only_hv, color=['AREG', 'PRLR', 'PGR', 'ESR1', 'FOXA1', 'TTF1', 'TFF3', 'SYTL2', 'AKT1', 'IGF1R'], ncols=4)


# In[ ]:


sc.pl.umap(KN23_only_hv, color = ['leiden'], frameon = False, legend_loc = "on data")


# In[22]:


sc.tl.rank_genes_groups(KN23_only_hv, 'leiden')
sc.pl.rank_genes_groups(KN23_only_hv, n_genes=20, sharey=False)


# In[23]:


markers_KN23 = sc.get.rank_genes_groups_df(KN23_only_hv, None)
markers_KN23 = markers_KN23[(markers_KN23.pvals_adj < 0.05) & (markers_KN23.logfoldchanges > .5)]
markers_KN23


# In[24]:


markers_KN23 = sc.get.rank_genes_groups_df(KN23_only_hv, None)


# In[26]:


markers_KN23[markers_KN23.names== 'ACTA2']


# ### Annotation

# In[ ]:


marker_genes = {
    "Epithelial": ['EPCAM'],
    "Luminal":['KRT19'],
    "Basal": ['TAGLN', 'KRT14', 'ACTA2', 'KRT17', 'SAA1', 'MYLK', 'KRT5', 'TP63'], 
    "Luminal_Mature": ['FOXA1', 'ESR1', 'AREG', 'MUCL1', 'PIP', 'AGR2'], 
    "Luminal_Progenitor": ['KRT15', 'LTF', 'SLPI', 'ELF5', 'EHF', 'PROM1', 'GABRP'], 
    "Adipocyte": ['APOE', 'ADIPOQ'], 
    "Endothelial": ['FABP5', 'MECOM', 'PECAM1', 'CLDN5', 'CDH5', 'FABP4'], 
    "Fibroblast": ['APOD', 'COL1A1', 'TNFAIP6', 'DCN', 'COL1A2'], 
    "General_Myeloid": ['HLA-DRA', 'HLA-DPA1', 'CD74'],
    "Monocyte": ['VCAN', 'CD14'],
    "Macrophage": ['APOE', 'CCL3', 'CCL4', 'IL1B', 'IDO1'], 
    "T-Cell": ['CCL5', 'CXCR4', 'CD2', 'GNLY', 'IL7R', 'PTPRC'], 
    "B-Cell": ['CD79B', 'IGKC'] 
}

sc.pl.dotplot(
    KN23_only,
    var_names=marker_genes,
    groupby='leiden'
)


# In[ ]:


sc.tl.rank_genes_groups(KN23_only, 'leiden')
markers = sc.get.rank_genes_groups_df(KN23_only, None)
markers = markers[(markers.pvals_adj < 0.05) & (markers.logfoldchanges > .5)]
markers


# In[ ]:


sc.pl.rank_genes_groups(KN23_only, n_genes=20, sharey=False)


# In[27]:


cell_type_KN23 = {"0":"Basal",
"1":"Fibroblast",
"2":"Luminal Mature",
"3":"Endothelial",
"4":"Luminal Progenitor",
"5":"Pericyte",
"6":"Fibroblast",
"7":"Macrophage",
"8":"Basal",
"9":"Basal",
"10":"Monocyte",
"11":"Pericyte",
"12":"Fibroblast",
                  "13":"Luminal Progenitor", "14":"Fibroblast", "15":"T-Cell"}


# In[28]:


KN23_only_hv.obs['cell type'] = KN23_only_hv.obs.leiden.map(cell_type_KN23)
sc.pl.umap(KN23_only_hv, color = ['cell type'], frameon = False)


# In[29]:


KN23_basal = KN23_only_hv[KN23_only_hv.obs['cell type'].isin(['Basal'])]
sc.pl.umap(KN23_basal, color = ['cell type'], frameon = False)


# In[30]:


KN23_mature_luminal = KN23_only_hv[KN23_only_hv.obs['cell type'].isin(['Luminal Mature'])]
sc.pl.umap(KN23_mature_luminal, color = ['cell type'], frameon = False)


# In[31]:


KN23_luminal_progenitor = KN23_only_hv[KN23_only_hv.obs['cell type'].isin(['Luminal Progenitor'])]
sc.pl.umap(KN23_luminal_progenitor, color = ['cell type'], frameon = False)


# In[188]:


KN23_epithelial = KN23_only_hv[KN23_only_hv.obs['cell type'].isin(['Basal', 'Luminal Mature', 'Luminal Progenitor'])]
sc.pl.umap(KN23_epithelial, color = ['cell type'], frameon = False)


# In[33]:


KN23_only_hv.write("./data/12_08_2025_KN23_only_hv_annotated.h5ad")


# In[34]:


KN23_basal.write("./data/12_08_2025_KN23_basal_hv_annotated.h5ad")
KN23_luminal_progenitor.write("./data/12_08_2025_KN23_luminal_progenitor_hv_annotated.h5ad")
KN23_mature_luminal.write("./data/12_08_2025_KN23_mature_luminal_hv_annotated.h5ad")


# In[178]:


KN23_only_hv = sc.read_h5ad("./data/12_08_2025_KN23_only_hv_annotated.h5ad")
print(KN23_only_hv)
print(KN23_only_hv.obs['cell type'].head())
sc.pl.umap(KN23_only_hv, color=['cell type'])


# In[14]:


KN23_only_hv = sc.read_h5ad("./data/12_08_2025_KN23_only_hv_annotated.h5ad")


# In[17]:


sc.pl.umap(KN23_only_hv, color = ['MME', 'cell type'], frameon = False)


# In[179]:


KN23_only_hv.obs['cell type'].value_counts()


# ### Basal

# In[35]:


KN23_basal = sc.read_h5ad("./data/12_08_2025_KN23_basal_hv_annotated.h5ad")


# In[52]:


sc.pp.highly_variable_genes(KN23_basal, n_top_genes = 2000)
KN23_basal_hv = KN23_basal[:, KN23_basal.var['highly_variable']].copy()
sc.pp.pca(KN23_basal_hv, n_comps=50)
sc.pp.neighbors(KN23_basal_hv, n_pcs = 20)
sc.tl.leiden(KN23_basal_hv, resolution = 0.125)
sc.tl.umap(KN23_basal_hv)


# In[53]:


sc.pl.umap(KN23_basal_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[54]:


sc.pl.umap(KN23_basal_hv, color=['ACTA2', 'TAGLN', 'MYL9', 'TPM2', 'ACTG2', 'KRT5', 'ITGA6', 'KRT14', 'KRT17', 'CCND2', 'DKK3', 'SPARC'], ncols=4)


# In[52]:


basal_leiden_KN23 = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4","5":"5"}
KN23_basal_hv.obs['subcluster'] = KN23_basal_hv.obs.leiden.map(basal_leiden_KN23)


# In[55]:


KN23_basal = sc.read_h5ad("./data/12_08_2025_KN23_basal_hv_annotated.h5ad")
KN23_basal.obs['subcluster'].value_counts()


# In[56]:


save_top_marker_genes(KN23_basal_hv, 'KN23', 'basal')


# In[54]:


KN23_basal_hv.write("./data/12_08_2025_KN23_basal_hv_annotated.h5ad")


# ### Luminal Progenitor

# In[57]:


KN23_luminal_progenitor = sc.read_h5ad("./data/12_08_2025_KN23_luminal_progenitor_hv_annotated.h5ad")


# In[60]:


sc.pp.highly_variable_genes(KN23_luminal_progenitor, n_top_genes = 2000)
KN23_luminal_progenitor_hv = KN23_luminal_progenitor[:, KN23_luminal_progenitor.var['highly_variable']].copy()
sc.pp.pca(KN23_luminal_progenitor_hv, n_comps=50)
sc.pp.neighbors(KN23_luminal_progenitor_hv, n_pcs = 20)
sc.tl.leiden(KN23_luminal_progenitor_hv, resolution = 0.2)
sc.tl.umap(KN23_luminal_progenitor_hv)
sc.pl.umap(KN23_luminal_progenitor_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[62]:


sc.pl.umap(KN23_luminal_progenitor_hv, color=['LTF', 'SLPI', 'RARRES1', 'ALDH1A3', 'KRT15', 'PI3', 'S100A9', 'CD24', 'KIT'], ncols=4)


# In[57]:


luminal_progenitor_leiden_KN23 = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4"}
KN23_luminal_progenitor_hv.obs['subcluster'] = KN23_luminal_progenitor_hv.obs.leiden.map(luminal_progenitor_leiden_KN23)


# In[64]:


save_top_marker_genes(KN23_luminal_progenitor_hv, 'KN23', 'luminal_progenitor')


# In[58]:


KN23_luminal_progenitor_hv.write("./data/12_08_2025_KN23_luminal_progenitor_hv_annotated.h5ad")


# In[59]:


KN23_luminal_progenitor_hv = sc.read_h5ad("./data/12_08_2025_KN23_luminal_progenitor_hv_annotated.h5ad")
KN23_luminal_progenitor_hv.obs['subcluster'].value_counts()


# ### Mature Luminal

# In[67]:


KN23_mature_luminal = sc.read_h5ad("./data/12_08_2025_KN23_mature_luminal_hv_annotated.h5ad")


# In[69]:


sc.pp.highly_variable_genes(KN23_mature_luminal, n_top_genes = 2000)
KN23_mature_luminal_hv = KN23_mature_luminal[:, KN23_mature_luminal.var['highly_variable']].copy()
sc.pp.pca(KN23_mature_luminal_hv, n_comps=50)
sc.pp.neighbors(KN23_mature_luminal_hv, n_pcs = 20)
sc.tl.leiden(KN23_mature_luminal_hv, resolution = 0.15)
sc.tl.umap(KN23_mature_luminal_hv)
sc.pl.umap(KN23_mature_luminal_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[70]:


sc.pl.umap(KN23_mature_luminal_hv, color=['LTF', 'SLPI', 'RARRES1', 'ALDH1A3', 'KRT15', 'PI3', 'S100A9', 'CD24', 'KIT'], ncols=4)


# In[61]:


mature_luminal_leiden_KN23 = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4"}
KN23_mature_luminal_hv.obs['subcluster'] = KN23_mature_luminal_hv.obs.leiden.map(mature_luminal_leiden_KN23)


# In[72]:


save_top_marker_genes(KN23_mature_luminal_hv, 'KN23', 'mature_luminal')


# In[62]:


KN23_mature_luminal_hv.write("./data/12_08_2025_KN23_mature_luminal_hv_annotated.h5ad")


# In[63]:


KN23_mature_luminal_hv = sc.read_h5ad("./data/12_08_2025_KN23_mature_luminal_hv_annotated.h5ad")
KN23_mature_luminal_hv.obs['subcluster'].value_counts()


# # N2021 Analysed Alone

# In[11]:


N2021_only = sc.concat([globals()[d] for d in dataobjects_N2021])


# In[12]:


print(N2021_only.shape)
sc.pp.filter_genes(N2021_only, min_cells = 200)
N2021_only.X = csr_matrix(N2021_only.X)
print(N2021_only.shape)


# In[13]:


N2021_only.obs.groupby('Sample').count()


# In[ ]:


#Create a dictionary mapping samples to mutation status
mutation_dict = {sample: "BRCA1" for sample in BRCA1}
mutation_dict.update({sample: "BRCA2" for sample in BRCA2})
mutation_dict.update({sample: "WT" for sample in WT})
mutation_dict.update({sample: "Unknown" for sample in Unknown})

#Assign Mutation Status to the AnnData object
N2021_only.obs["Mutation_Status"] = N2021_only.obs["Sample"].map(mutation_dict).fillna("Unknown")


# In[ ]:


print(N2021_only.obs[["Batch", "Sample", "Mutation_Status"]])


# In[14]:


N2021_only.layers['counts'] = N2021_only.X.copy()
sc.pp.normalize_total(N2021_only, target_sum = 1e4)
sc.pp.log1p(N2021_only)
N2021_only.raw = N2021_only
N2021_only.obs.head()


# In[15]:


sc.pp.highly_variable_genes(N2021_only, n_top_genes = 2000)
N2021_only_hv = N2021_only[:, N2021_only.var['highly_variable']].copy()
sc.pp.pca(N2021_only_hv, n_comps=50)
sc.pp.neighbors(N2021_only_hv, n_pcs = 20)
sc.tl.leiden(N2021_only_hv, resolution = 0.4)
sc.tl.umap(N2021_only_hv)
sc.pl.umap(N2021_only_hv, color=['leiden', 'Batch', 'Sample'])


# In[ ]:


sc.pl.pca_variance_ratio(N2021_only_hv, log=True)  #Scree plot


# In[16]:


#BASAL Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(N2021_only_hv, color=['ACTA2', 'TAGLN', 'MYL9', 'TPM2', 'ACTG2', 'KRT5', 'ITGA6', 'KRT14', 'KRT17', 'CCND2', 'DKK3', 'SPARC'], ncols=4)


# In[17]:


#LUMINAL PROGENITOR Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(N2021_only_hv, color=['LTF', 'SLPI', 'RARRES1', 'ALDH1A3', 'KRT15', 'PI3', 'S100A9', 'CD24', 'KIT'], ncols=4)


# In[18]:


#MATURE LUMINAL Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(N2021_only_hv, color=['AREG', 'PRLR', 'PGR', 'ESR1', 'FOXA1', 'TTF1', 'TFF3', 'SYTL2', 'AKT1', 'IGF1R'], ncols=4)


# In[19]:


sc.pl.umap(N2021_only_hv, color = ['leiden'], frameon = False, legend_loc = "on data")


# In[ ]:


#HBCA# BASAL #
sc.pl.umap(N2021_only_hv, color=['KRT5','KRT14', 'KRT17','TAGLN','TPM2','KRT17','MT2A','DST'])


# In[ ]:


#HPA Basal#
sc.pl.umap(N2021_only_hv, color=['ITGA6','PLPPR3'])


# In[ ]:


#HBCA# LumHR (ALL)= ?
sc.pl.umap(N2021_only_hv, color=['EPCAM','KRT18'])


# In[ ]:


#HBCA# LumHR (CLUSTERS)= ?
sc.pl.umap(N2021_only_hv, color=['KRT18','EGLN3','FASN'])


# In[ ]:


#HBCA# LumSec (ALL)=?
sc.pl.umap(N2021_only_hv, color=['KRT15','KRT16','KRT7'])


# In[ ]:


#HBCA# LumSec (CLUSTERS)=?
sc.pl.umap(N2021_only_hv, color=['KRT15','KRT16','KRT7','PTN','CD74','KIT','ACTA2'])


# ## Annotating  

# In[ ]:


N2021_only_hv = sc.read_h5ad("./data/12_08_2025_N2021_only_hv_annotated.h5ad")


# In[ ]:


marker_genes = {
    "Epithelial": ['EPCAM'],
    "Luminal":['KRT19'],
    "Basal": ['TAGLN', 'KRT14', 'ACTA2', 'KRT17', 'SAA1', 'MYLK', 'KRT5', 'TP63'], 
    "Luminal_Mature": ['FOXA1', 'ESR1', 'AREG', 'MUCL1', 'PIP', 'AGR2'], 
    "Luminal_Progenitor": ['KRT15', 'LTF', 'SLPI', 'ELF5', 'EHF', 'PROM1', 'GABRP'], 
    "Adipocyte": ['APOE', 'ADIPOQ'], 
    "Endothelial": ['FABP5', 'MECOM', 'PECAM1', 'CLDN5', 'CDH5', 'FABP4'], 
    "Fibroblast": ['APOD', 'COL1A1', 'TNFAIP6', 'DCN', 'COL1A2'], 
    "General_Myeloid": ['HLA-DRA', 'HLA-DPA1', 'CD74'],
    "Monocyte": ['VCAN', 'CD14'],
    "Macrophage": ['APOE', 'CCL3', 'CCL4', 'IL1B', 'IDO1'], 
    "T-Cell": ['CCL5', 'CXCR4', 'CD2', 'GNLY', 'IL7R', 'PTPRC'], 
    "B-Cell": ['CD79B', 'IGKC'] 
}

sc.pl.dotplot(
    N2021_only_hv,
    var_names=marker_genes,
    groupby='leiden'
)


# In[40]:


cell_type_N2021 = {"0":"Luminal Progenitor",
"1":"Endothelial",
"2":"Luminal Mature",
"3":"Adipocyte",
"4":"Endothelial",
"5":"Luminal Mature",
"6":"Luminal Mature",
"7":"Fibroblast",
"8":"Luminal Mature",
"9":"T-Cell",
"10":"Luminal Progenitor",
"11":"Basal",
"12":"Fibroblast",
                  "13":"Luminal Progenitor"}


# In[41]:


N2021_only_hv.obs['cell type'] = N2021_only_hv.obs.leiden.map(cell_type_N2021)


# In[42]:


sc.pl.umap(N2021_only_hv, color = ['cell type'], frameon = False)


# In[6]:


N2021_basal = N2021_only_hv[N2021_only_hv.obs['cell type'].isin(['Basal'])]
sc.pl.umap(N2021_basal, color = ['cell type'], frameon = False)


# In[7]:


N2021_mature_luminal = N2021_only_hv[N2021_only_hv.obs['cell type'].isin(['Luminal Mature'])]
sc.pl.umap(N2021_mature_luminal, color = ['cell type'], frameon = False)


# In[8]:


N2021_luminal_progenitor = N2021_only_hv[N2021_only_hv.obs['cell type'].isin(['Luminal Progenitor'])]
sc.pl.umap(N2021_luminal_progenitor, color = ['cell type'], frameon = False)


# In[189]:


N2021_epithelial = N2021_only_hv[N2021_only_hv.obs['cell type'].isin(['Basal', 'Luminal Mature', 'Luminal Progenitor'])]
sc.pl.umap(N2021_epithelial, color = ['cell type'], frameon = False)


# In[46]:


N2021_only_hv.write("./data/12_08_2025_N2021_only_hv_annotated.h5ad")


# In[176]:


N2021_only_hv = sc.read_h5ad("./data/12_08_2025_N2021_only_hv_annotated.h5ad")
print(N2021_only_hv)
print(N2021_only_hv.obs['cell type'].head())
sc.pl.umap(N2021_only_hv, color=['cell type'])


# In[18]:


N2021_only_hv = sc.read_h5ad("./data/12_08_2025_N2021_only_hv_annotated.h5ad")
sc.pl.umap(N2021_only_hv, color = ['MME', 'cell type'], frameon = False)


# In[177]:


N2021_only_hv.obs['cell type'].value_counts()


# ### Markers N2021

# In[20]:


sc.tl.rank_genes_groups(N2021_only_hv, 'leiden')

sc.pl.rank_genes_groups(N2021_only_hv, n_genes=20, sharey=False)


# In[21]:


markers_N2021 = sc.get.rank_genes_groups_df(N2021_only_hv, None)
markers_N2021 = markers_N2021[(markers_N2021.pvals_adj < 0.05) & (markers_N2021.logfoldchanges > .5)]
markers_N2021


# In[22]:


markers_N2021 = sc.get.rank_genes_groups_df(N2021_only_hv, None)


# In[23]:


#ali: BASAL#
markers_N2021[markers_N2021.names== 'ACTA2']


# In[24]:


markers_N2021[markers_N2021.names== 'TPM2']


# In[25]:


markers_N2021[markers_N2021.names== 'KRT14']


# In[26]:


markers_N2021[markers_N2021.names== 'MYL9']


# In[27]:


markers_N2021[markers_N2021.names== 'TAGLN']


# In[28]:


markers_N2021[markers_N2021.names== 'KRT14']


# In[29]:


#Luminal Progenitor
markers_N2021[markers_N2021.names== 'LTF']


# In[30]:


#Luminal Progenitor
markers_N2021[markers_N2021.names== 'SLPI']


# In[31]:


#Luminal Progenitor
markers_N2021[markers_N2021.names== 'KRT15']


# In[32]:


#Luminal Progenitor
markers_N2021[markers_N2021.names== 'KIT']


# In[33]:


#Luminal Progenitor
markers_N2021[markers_N2021.names== 'RARRES1']


# In[34]:


#Luminal Progenitor
markers_N2021[markers_N2021.names== 'ALDH1A3']


# In[35]:


#Luminal Mature
markers_N2021[markers_N2021.names== 'AREG']


# In[36]:


#Luminal Mature
markers_N2021[markers_N2021.names== 'FOXA1']


# In[37]:


#Luminal Mature
markers_N2021[markers_N2021.names== 'TFF3']


# In[38]:


#Luminal Mature
markers_N2021[markers_N2021.names== 'ESR1']


# ### Basal N2021 Clustering

# In[75]:


N2021_basal = sc.read_h5ad("./data/17_10_2025_N2021_basal_hv_annotated.h5ad")


# In[76]:


sc.pp.highly_variable_genes(N2021_basal, n_top_genes = 2000)
N2021_basal_hv = N2021_basal[:, N2021_basal.var['highly_variable']].copy()
sc.pp.pca(N2021_basal, n_comps=50)
sc.pp.neighbors(N2021_basal_hv, n_pcs = 20)


# In[9]:


sc.tl.leiden(N2021_basal_hv, key_added="leiden_res0_25", resolution=0.25)
sc.tl.leiden(N2021_basal_hv, key_added="leiden_res0_5", resolution=0.5)
sc.tl.leiden(N2021_basal_hv, key_added="leiden_res0_1", resolution=0.1)
sc.tl.leiden(N2021_basal_hv, key_added="leiden_res0_2", resolution=0.2)
sc.tl.leiden(N2021_basal_hv, key_added="leiden_res0_05", resolution=0.05)
sc.tl.leiden(N2021_basal_hv, key_added="leiden_res0_15", resolution=0.15)
sc.tl.dendrogram(N2021_basal_hv, groupby = "leiden_res0_25")
sc.pl.dendrogram(N2021_basal_hv, groupby = "leiden_res0_25")

sc.tl.dendrogram(N2021_basal_hv, groupby = "leiden_res0_5")
sc.pl.dendrogram(N2021_basal_hv, groupby = "leiden_res0_5")

sc.tl.dendrogram(N2021_basal_hv, groupby = "leiden_res0_1")
sc.pl.dendrogram(N2021_basal_hv, groupby = "leiden_res0_1")

sc.tl.dendrogram(N2021_basal_hv, groupby = "leiden_res0_2")
sc.pl.dendrogram(N2021_basal_hv, groupby = "leiden_res0_2")

sc.tl.dendrogram(N2021_basal_hv, groupby = "leiden_res0_05")
sc.pl.dendrogram(N2021_basal_hv, groupby = "leiden_res0_05")

sc.tl.dendrogram(N2021_basal_hv, groupby = "leiden_res0_15")
sc.pl.dendrogram(N2021_basal_hv, groupby = "leiden_res0_15")


# In[6]:


sc.pl.umap(N2021_basal_hv, color=["leiden_res0_25", "leiden_res0_5"], legend_loc="on data")


# In[77]:


sc.pp.highly_variable_genes(N2021_basal, n_top_genes = 2000)
N2021_basal_hv = N2021_basal[:, N2021_basal.var['highly_variable']].copy()
sc.pp.pca(N2021_basal, n_comps=50)
sc.pp.neighbors(N2021_basal_hv, n_pcs = 20)
sc.tl.leiden(N2021_basal_hv, resolution = 0.25)
sc.tl.umap(N2021_basal_hv)


# In[20]:


#Mixed Epithelial Cluster: Basal vs Luminal. Conclusion: Mixed Epithelial Cluster is Luminal
#sc.pl.umap(N2021_basal_hv, color=['CD24'])


# In[78]:


sc.pl.umap(N2021_basal_hv, color=['ACTA2', 'TAGLN', 'MYL9', 'TPM2', 'ACTG2', 'KRT5', 'ITGA6', 'KRT14', 'KRT17', 'CCND2', 'DKK3', 'SPARC'], ncols=4)


# In[79]:


sc.pl.umap(N2021_basal_hv, color=['cell type'], ncols=2)


# In[80]:


sc.pl.umap(N2021_basal_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[81]:


basal_leiden_N2021 = {"0":"0", "1":"1", "2":"2"}
N2021_basal_hv.obs['subcluster'] = N2021_basal_hv.obs.leiden.map(basal_leiden_N2021)


# In[82]:


N2021_basal_hv.write("./data/17_10_2025_N2021_basal_hv_annotated.h5ad")


# In[46]:


N2021_basal = sc.read_h5ad("./data/17_10_2025_N2021_basal_hv_annotated.h5ad")


# In[48]:


N2021_basal.obs['subcluster'].value_counts()


# In[75]:


#N2021_basal_only_hv = N2021_basal_hv[N2021_basal_hv.obs['leiden'].isin(['0','1','2','4'])]
#sc.pl.umap(N2021_basal_only_hv, color=['leiden'])


# In[83]:


sc.tl.rank_genes_groups(N2021_basal_hv, 'leiden')

sc.pl.rank_genes_groups(N2021_basal_hv, n_genes=20, sharey=False)


# In[95]:


import os
#add gene_counts 20, 30, 100 back later
def save_top_marker_genes(adata, study_name, cell_type, gene_counts=[20, 30, 100, 1000]):
    """
    Saves the top differentially expressed marker genes for each subcluster to CSV files.
    
    Parameters:
    - adata: AnnData object
    - study_name: str, the name of the study (e.g., 'N2021')
    - cell_type: str, the cell type (e.g., 'basal')
    - gene_counts: list, number of top genes to save (default: [20, 30, 100])
    """
    # Create the output directory if it doesn't exist
    output_dir = f"{cell_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run differential expression analysis
    sc.tl.rank_genes_groups(adata, 'leiden')
    
    # Iterate through each leiden cluster
    for group in adata.obs['leiden'].unique():
        for n_genes in gene_counts:
            # Get top differentially expressed genes
            top_genes_df = sc.get.rank_genes_groups_df(adata, group=group)[:n_genes]
            
            # Define filename
            filename = f"{output_dir}/up{n_genes}_{study_name}_{cell_type}_{group}.csv"
            
            # Save to CSV
            top_genes_df.to_csv(filename, index=False)
            print(f"Saved: {filename}")


# In[84]:


save_top_marker_genes(N2021_basal_hv, 'N2021', 'basal')


# In[ ]:


#new method: pairwise comparison 
import os
import numpy as np
import pandas as pd
import scanpy as sc

def find_pairwise_markers(adata, study_title, cell_type, leiden_res='leiden', top_n=[20, 30, 100], initial_top=300):
    clusters = adata.obs[leiden_res].unique()
    all_results = {}
    output_dir = "N2021_Basal"  # New folder for saving results
    os.makedirs(output_dir, exist_ok=True)  # Create folder if it doesn't exist

    for cluster in clusters:
        gene_counts = {}  # Count occurrences of genes
        logfc_values = {}  # Store log fold changes

        for other in clusters:
            if cluster != other:
                sc.tl.rank_genes_groups(adata, groupby=leiden_res, groups=[cluster], reference=other, method='wilcoxon')
                result_df = sc.get.rank_genes_groups_df(adata, group=cluster)
                top_genes = result_df.nlargest(initial_top, 'logfoldchanges')

                for gene, logfc in zip(top_genes['names'], top_genes['logfoldchanges']):
                    gene_counts[gene] = gene_counts.get(gene, 0) + 1
                    if gene in logfc_values:
                        logfc_values[gene].append(logfc)
                    else:
                        logfc_values[gene] = [logfc]

        # Change: Filter genes appearing in more than 1 comparison
        filtered_genes = {gene: np.mean(logfcs) for gene, logfcs in logfc_values.items() if gene_counts[gene] > 1}

        # Sort by mean log fold change
        sorted_genes = sorted(filtered_genes.items(), key=lambda x: x[1], reverse=True)
        df = pd.DataFrame(sorted_genes, columns=['Gene', 'Mean_LogFC'])

        # Save top 20, 30, 100 as snapshots in "N2021" folder
        for n in top_n:
            snapshot_df = df.head(n)
            filename = f"{output_dir}/{study_title}_{cell_type}_cluster{cluster}_up{n}.csv"
            snapshot_df.to_csv(filename, index=False)
            print(f"Saved: {filename}")

print("\n### Process Complete! All results saved in 'N2021' folder. ###")


# In[ ]:


find_pairwise_markers(N2021_basal_hv, study_title="N2021", cell_type="basal", leiden_res='leiden', top_n=[20, 30, 100])


# In[92]:


#validation of find_pairwise_markers function to print all intermediate steps as csv
import os
import numpy as np
import pandas as pd
import scanpy as sc

def validate_find_pairwise_markers(adata, study_title, cell_type, leiden_res='leiden', top_n=[20, 30, 100], initial_top=300):
    clusters = adata.obs[leiden_res].unique()
    output_dir = f"N2021"  # Main output folder
    test_output_dir = f"Test_{output_dir}"  # Validation folder
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    for cluster in clusters:
        gene_counts = {}  # Count occurrences of genes
        logfc_values = {}  # Store log fold changes
        raw_rankings = []  # Store raw rankings for validation

        for other in clusters:
            if cluster != other:
                sc.tl.rank_genes_groups(adata, groupby=leiden_res, groups=[cluster], reference=other, method='wilcoxon')
                result_df = sc.get.rank_genes_groups_df(adata, group=cluster)
                top_genes = result_df.nlargest(initial_top, 'logfoldchanges')

                # Store raw rankings for validation
                ranking_filename = f"{test_output_dir}/{study_title}_{cell_type}_cluster{cluster}_vs_{other}_raw.csv"
                top_genes.to_csv(ranking_filename, index=False)
                print(f"Saved raw ranking: {ranking_filename}")

                for gene, logfc in zip(top_genes['names'], top_genes['logfoldchanges']):
                    gene_counts[gene] = gene_counts.get(gene, 0) + 1
                    if gene in logfc_values:
                        logfc_values[gene].append(logfc)
                    else:
                        logfc_values[gene] = [logfc]

        # Save unfiltered genes (before applying the filtering rule)
        unfiltered_filename = f"{test_output_dir}/{study_title}_{cell_type}_cluster{cluster}_unfiltered.csv"
        pd.DataFrame(gene_counts.items(), columns=['Gene', 'Count']).to_csv(unfiltered_filename, index=False)
        print(f"Saved unfiltered genes: {unfiltered_filename}")

        # Filter genes appearing in more than 1 comparison
        filtered_genes = {gene: np.mean(logfcs) for gene, logfcs in logfc_values.items() if gene_counts[gene] > 1}

        # Save filtered genes (after applying the filtering rule)
        filtered_filename = f"{test_output_dir}/{study_title}_{cell_type}_cluster{cluster}_filtered.csv"
        pd.DataFrame(filtered_genes.items(), columns=['Gene', 'Mean_LogFC']).to_csv(filtered_filename, index=False)
        print(f"Saved filtered genes: {filtered_filename}")

        # Sort by mean log fold change
        sorted_genes = sorted(filtered_genes.items(), key=lambda x: x[1], reverse=True)
        df = pd.DataFrame(sorted_genes, columns=['Gene', 'Mean_LogFC'])

        # Save top 20, 30, 100 as snapshots in "N2021" folder
        for n in top_n:
            snapshot_df = df.head(n)
            filename = f"{output_dir}/{study_title}_{cell_type}_cluster{cluster}_up{n}.csv"
            snapshot_df.to_csv(filename, index=False)
            print(f"Saved: {filename}")

    print("\n### Process Complete! All results saved in 'N2021' and validation files in 'Test_N2021'. ###")


# In[93]:


validate_find_pairwise_markers(N2021_basal_hv, study_title="N2021", cell_type="basal", leiden_res='leiden', top_n=[20, 30, 100])


# In[94]:


#compare to scanpy function get_rank_genes, but instead of ordering top
#genes by score, order by log fold change (so can compare new & old method)

sc.tl.rank_genes_groups(N2021_basal_hv, 'leiden')

# Retrieve the ranked genes and explicitly sort by logfoldchanges
up30_N2021_basal_0_results = sc.get.rank_genes_groups_df(N2021_basal_hv, group='0')
up30_N2021_basal_0_results = up30_N2021_basal_0_results.sort_values(by='logfoldchanges', ascending=False).head(30)

# Save the sorted results
up30_N2021_basal_0_results.to_csv('scanpy_lfc_up30_N2021_basal_0_results.csv', index=False)


# ### Luminal Mature N2021 Clustering

# In[85]:


N2021_mature_luminal = sc.read_h5ad("./data/17_10_2025_N2021_mature_luminal_hv_annotated.h5ad")


# In[86]:


sc.pp.highly_variable_genes(N2021_mature_luminal, n_top_genes = 2000)
N2021_mature_luminal_hv = N2021_mature_luminal[:, N2021_mature_luminal.var['highly_variable']].copy()
sc.pp.pca(N2021_mature_luminal_hv)
sc.pp.neighbors(N2021_mature_luminal_hv, n_pcs = 20)
sc.tl.leiden(N2021_mature_luminal_hv, resolution = 0.25)
sc.tl.umap(N2021_mature_luminal_hv)


# In[87]:


sc.tl.leiden(N2021_mature_luminal_hv, key_added="leiden_res0_25", resolution=0.25)
sc.tl.leiden(N2021_mature_luminal_hv, key_added="leiden_res0_5", resolution=0.5)
sc.tl.leiden(N2021_mature_luminal_hv, key_added="leiden_res0_1", resolution=0.1)
sc.tl.leiden(N2021_mature_luminal_hv, key_added="leiden_res0_2", resolution=0.2)
sc.tl.leiden(N2021_mature_luminal_hv, key_added="leiden_res0_05", resolution=0.05)
sc.tl.leiden(N2021_mature_luminal_hv, key_added="leiden_res0_15", resolution=0.15)


# In[88]:


sc.pl.umap(N2021_mature_luminal_hv, color=["leiden_res0_5", "leiden_res0_25", "leiden_res0_2", "leiden_res0_15", "leiden_res0_1", "leiden_res0_05"], legend_loc="on data")


# In[89]:


sc.pl.umap(N2021_mature_luminal_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[90]:


ml_leiden_N2021 = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4"}
N2021_mature_luminal_hv.obs['subcluster'] = N2021_mature_luminal_hv.obs.leiden.map(ml_leiden_N2021)

N2021_mature_luminal_hv.write("./data/17_10_2025_N2021_mature_luminal_hv_annotated.h5ad")


# In[49]:


N2021_mature_luminal = sc.read_h5ad("./data/17_10_2025_N2021_mature_luminal_hv_annotated.h5ad")
N2021_mature_luminal.obs['subcluster'].value_counts()


# In[91]:


sc.tl.rank_genes_groups(N2021_mature_luminal_hv, 'leiden')

sc.pl.rank_genes_groups(N2021_mature_luminal_hv, n_genes=20, sharey=False)


# In[92]:


save_top_marker_genes(N2021_mature_luminal_hv, 'N2021', 'mature_luminal')


# ### Luminal Progenitor N2021 Clustering

# In[93]:


N2021_luminal_progenitor = sc.read_h5ad("./data/17_10_2025_N2021_luminal_progenitor_hv_annotated.h5ad")


# In[98]:


sc.pp.highly_variable_genes(N2021_luminal_progenitor, n_top_genes = 2000)
N2021_luminal_progenitor_hv = N2021_luminal_progenitor[:, N2021_luminal_progenitor.var['highly_variable']].copy()
sc.pp.pca(N2021_luminal_progenitor_hv)
sc.pp.neighbors(N2021_luminal_progenitor_hv, n_pcs = 20)
sc.tl.leiden(N2021_luminal_progenitor_hv, resolution = 0.25)
sc.tl.umap(N2021_luminal_progenitor_hv)


# In[99]:


sc.pl.umap(N2021_luminal_progenitor_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[100]:


lp_leiden_N2021 = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4", "5":"5", "6":"6", "7":"7"}
N2021_luminal_progenitor_hv.obs['subcluster'] = N2021_luminal_progenitor_hv.obs.leiden.map(lp_leiden_N2021)

N2021_luminal_progenitor_hv.write("./data/17_10_2025_N2021_luminal_progenitor_hv_annotated.h5ad")


# In[50]:


N2021_luminal_progenitor = sc.read_h5ad("./data/17_10_2025_N2021_luminal_progenitor_hv_annotated.h5ad")
N2021_luminal_progenitor.obs['subcluster'].value_counts()


# In[101]:


#LUMINAL PROGENITOR Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(N2021_luminal_progenitor_hv, color=['LTF', 'SLPI', 'RARRES1', 'ALDH1A3', 'KRT15', 'PI3', 'S100A9', 'CD24', 'KIT'], ncols=4)


# In[102]:


sc.tl.rank_genes_groups(N2021_luminal_progenitor_hv, 'leiden')

sc.pl.rank_genes_groups(N2021_luminal_progenitor_hv, n_genes=20, sharey=False)


# In[103]:


save_top_marker_genes(N2021_luminal_progenitor_hv, 'N2021', 'luminal_progenitor')


# # NN2023 Analysed Alone #

# In[82]:


#NN2023_only = sc.concat([globals()[d] for d in inds_NN2023_data])
NN2023_only = inds_NN2023_data


# In[83]:


print(NN2023_only.shape)
sc.pp.filter_genes(NN2023_only, min_cells = 200)
NN2023_only.X = csr_matrix(NN2023_only.X)
print(NN2023_only.shape)


# In[84]:


NN2023_only.var["mt"] = NN2023_only.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(
    NN2023_only, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)
sc.pl.scatter(NN2023_only, x="total_counts", y="pct_counts_mt")
sc.pl.scatter(NN2023_only, x="total_counts", y="n_genes_by_counts")


# In[85]:


NN2023_only1 = NN2023_only[NN2023_only.obs.n_genes_by_counts < 2500, :]
NN2023_only1 = NN2023_only[NN2023_only.obs.pct_counts_mt < 6, :].copy()
NN2023_only.obs.groupby('Sample').count()


# In[86]:


NN2023_only.layers['counts'] = NN2023_only.X.copy()
sc.pp.normalize_total(NN2023_only, target_sum = 1e4)
sc.pp.log1p(NN2023_only)
NN2023_only.raw = NN2023_only
NN2023_only.obs.head()


# In[91]:


sc.pp.highly_variable_genes(NN2023_only, n_top_genes = 2000)
NN2023_only_hv = NN2023_only[:, NN2023_only.var['highly_variable']].copy()
sc.pp.pca(NN2023_only_hv)
sc.pp.neighbors(NN2023_only_hv, n_pcs = 20)
sc.tl.leiden(NN2023_only_hv, resolution = 0.2)
sc.tl.umap(NN2023_only_hv)
sc.pl.umap(NN2023_only_hv, color=['leiden', 'Batch', 'Sample'])


# In[88]:


#BASAL Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(NN2023_only_hv, color=['ACTA2','TAGLN', 'MYL9', 'TPM2', 'ACTG2','KRT5', 'ITGA6', 'KRT14', 'KRT17', 'CCND2', 'DKK3', 'SPARC', 'KRT14'], gene_symbols='feature_name', ncols=4)


# In[89]:


#LUMINAL PROGENITOR Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(NN2023_only_hv, color=['LTF', 'SLPI', 'RARRES1', 'ALDH1A3', 'KRT15', 'PI3', 'S100A9', 'CD24', 'KIT'], ncols=4, gene_symbols='feature_name')


# In[90]:


#MATURE LUMINAL Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(NN2023_only_hv, color=['AREG', 'STC2','PRLR', 'PGR', 'ESR1', 'FOXA1', 'TTF1', 'TFF3', 'SYTL2', 'AKT1', 'IGF1R', 'ALCAM','STC2'], ncols=4, gene_symbols='feature_name')


# In[92]:


sc.pl.umap(NN2023_only_hv, color = ['leiden'], frameon = False, legend_loc = "on data")


# In[95]:


cell_type_NN2023 = {"0":"Basal",
"1":"Luminal Mature",
"2":"Luminal Progenitor",
"3":"Basal",
"4":"Luminal Mature",
"5":"Luminal Progenitor",
"6":"Luminal Mature",
"7":"Luminal Mature"}
NN2023_only_hv.obs['cell type'] = NN2023_only_hv.obs.leiden.map(cell_type_NN2023)
sc.pl.umap(NN2023_only_hv, color = ['cell type'], frameon = False)


# In[105]:


NN2023_basal = NN2023_only_hv[NN2023_only_hv.obs['cell type'].isin(['Basal'])]
sc.pl.umap(NN2023_basal, color = ['cell type'], frameon = False)


# In[106]:


NN2023_mature_luminal = NN2023_only_hv[NN2023_only_hv.obs['cell type'].isin(['Luminal Mature'])]
sc.pl.umap(NN2023_mature_luminal, color = ['cell type'], frameon = False)


# In[107]:


NN2023_luminal_progenitor = NN2023_only_hv[NN2023_only_hv.obs['cell type'].isin(['Luminal Progenitor'])]
sc.pl.umap(NN2023_luminal_progenitor, color = ['cell type'], frameon = False)


# In[190]:


NN2023_epithelial = NN2023_only_hv[NN2023_only_hv.obs['cell type'].isin(['Basal', 'Luminal Mature', 'Luminal Progenitor'])]
sc.pl.umap(NN2023_epithelial, color = ['cell type'], frameon = False)


# In[100]:


NN2023_only_hv.write("./data/12_08_2025_NN2023_only_hv_annotated.h5ad")


# In[173]:


NN2023_only_hv = sc.read_h5ad("./data/12_08_2025_NN2023_only_hv_annotated.h5ad")
print(NN2023_only_hv)
print(NN2023_only_hv.obs['cell type'].head())
sc.pl.umap(NN2023_only_hv, color=['cell type'])


# In[19]:


NN2023_only_hv = sc.read_h5ad("./data/12_08_2025_NN2023_only_hv_annotated.h5ad")
sc.pl.umap(NN2023_only_hv, color = ['MME', 'cell type'], frameon = False)


# In[108]:


NN2023_basal.write("./data/17_10_2025_NN2023_basal_hv_annotated.h5ad")
NN2023_mature_luminal.write("./data/17_10_2025_NN2023_mature_luminal_hv_annotated.h5ad")
NN2023_luminal_progenitor.write("./data/17_10_2025_NN2023_luminal_progenitor_hv_annotated.h5ad")


# In[174]:


NN2023_only_hv.obs['cell type'].value_counts()


# ### Markers ###

# In[93]:


sc.tl.rank_genes_groups(NN2023_only_hv, 'leiden')
sc.pl.rank_genes_groups(NN2023_only_hv, n_genes=20, sharey=False)


# In[94]:


markers_NN2023 = sc.get.rank_genes_groups_df(NN2023_only_hv, None)
markers_NN2023 = markers_NN2023[(markers_NN2023.pvals_adj < 0.05) & (markers_NN2023.logfoldchanges > .5)]
markers_NN2023


# ### Basal ###

# In[109]:


NN2023_basal_hv = sc.read_h5ad("./data/17_10_2025_NN2023_basal_hv_annotated.h5ad")


# In[110]:


#sc.pp.highly_variable_genes(NN2023_basal, n_top_genes = 2000)
#NN2023_basal_hv = NN2023_basal[:, NN2023_basal.var['highly_variable']].copy()
sc.pp.pca(NN2023_basal_hv)
sc.pp.neighbors(NN2023_basal_hv, n_pcs = 20)
sc.tl.leiden(NN2023_basal_hv, resolution = 0.1)
sc.tl.umap(NN2023_basal_hv)
sc.pl.umap(NN2023_basal_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[ ]:


sc.pl.umap(NN2023_basal_hv, color=['ACTA2', 'TAGLN', 'MYL9', 'TPM2', 'ACTG2', 'KRT5', 'ITGA6', 'KRT14', 'KRT17', 'CCND2', 'DKK3', 'SPARC'], ncols=4, gene_symbols='feature_name')


# In[43]:


basal_leiden_NN2023 = {"0":"0", "1":"1"}
NN2023_basal_hv.obs['subcluster'] = NN2023_basal_hv.obs.leiden.map(basal_leiden_NN2023)

NN2023_basal_hv.write("./data/17_10_2025_NN2023_basal_hv_annotated.h5ad")


# In[64]:


NN2023_basal_hv = sc.read_h5ad("./data/17_10_2025_NN2023_basal_hv_annotated.h5ad")
NN2023_basal_hv.obs['subcluster'].value_counts()


# In[16]:


save_top_marker_genes(NN2023_basal_hv, 'NN2023', 'basal')


# In[12]:


sc.tl.rank_genes_groups(NN2023_basal_hv, 'leiden')

sc.pl.rank_genes_groups(NN2023_basal_hv, n_genes=20, sharey=False)


# ### Luminal Prog ###

# In[ ]:


NN2023_luminal_progenitor_hv = sc.read_h5ad("./data/17_10_2025_NN2023_luminal_progenitor_hv_annotated.h5ad")


# In[41]:


#sc.pp.highly_variable_genes(NN2023_luminal_progenitor, n_top_genes = 2000)
#NN2023_luminal_progenitor_hv = NN2023_luminal_progenitor[:, NN2023_luminal_progenitor.var['highly_variable']].copy()
sc.pp.pca(NN2023_luminal_progenitor_hv)
sc.pp.neighbors(NN2023_luminal_progenitor_hv, n_pcs = 20)
sc.tl.leiden(NN2023_luminal_progenitor_hv, resolution = 0.1)
sc.tl.umap(NN2023_luminal_progenitor_hv)


# In[45]:


sc.pl.umap(NN2023_luminal_progenitor_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[18]:


#LUMINAL PROGENITOR Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(NN2023_luminal_progenitor_hv, color=['LTF', 'SLPI', 'RARRES1', 'ALDH1A3', 'KRT15', 'PI3', 'S100A9', 'CD24', 'KIT'], ncols=4, gene_symbols='feature_name')


# In[47]:


lp_leiden_NN2023 = {"0":"0", "1":"1", "2":"2"}
NN2023_luminal_progenitor_hv.obs['subcluster'] = NN2023_luminal_progenitor_hv.obs.leiden.map(lp_leiden_NN2023)

NN2023_luminal_progenitor_hv.write("./data/17_10_2025_NN2023_luminal_progenitor_hv_annotated.h5ad")


# In[65]:


NN2023_luminal_progenitor_hv = sc.read_h5ad("./data/17_10_2025_NN2023_luminal_progenitor_hv_annotated.h5ad")
NN2023_luminal_progenitor_hv.obs['subcluster'].value_counts()


# In[22]:


save_top_marker_genes(NN2023_basal_hv, 'NN2023', 'luminal_progenitor')


# In[88]:


sc.tl.rank_genes_groups(NN2023_luminal_progenitor_hv, 'leiden')

sc.pl.rank_genes_groups(NN2023_luminal_progenitor_hv, n_genes=20, sharey=False)


# ### Luminal Mature ###

# In[ ]:


NN2023_mature_luminal_hv = sc.read_h5ad("./data/17_10_2025_NN2023_mature_luminal_hv_annotated.h5ad")


# In[42]:


#sc.pp.highly_variable_genes(NN2023_mature_luminal, n_top_genes = 2000)
#NN2023_mature_luminal_hv = NN2023_mature_luminal[:, NN2023_mature_luminal.var['highly_variable']].copy()
sc.pp.pca(NN2023_mature_luminal_hv)
sc.pp.neighbors(NN2023_mature_luminal_hv, n_pcs = 20)
sc.tl.leiden(NN2023_mature_luminal_hv, resolution = 0.2)
sc.tl.umap(NN2023_mature_luminal_hv)


# In[46]:


sc.pl.umap(NN2023_mature_luminal_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[48]:


ml_leiden_NN2023 = {"0":"0", "1":"1", "2":"2", "3":"3"}
NN2023_mature_luminal_hv.obs['subcluster'] = NN2023_mature_luminal_hv.obs.leiden.map(ml_leiden_NN2023)

NN2023_mature_luminal_hv.write("./data/17_10_2025_NN2023_mature_luminal_hv_annotated.h5ad")


# In[66]:


NN2023_mature_luminal_hv = sc.read_h5ad("./data/17_10_2025_NN2023_mature_luminal_hv_annotated.h5ad")
NN2023_mature_luminal_hv.obs['subcluster'].value_counts()


# In[ ]:


sc.pl.umap(NN2023_mature_luminal_hv, color=['AREG', 'PRLR', 'PGR', 'ESR1', 'FOXA1', 'TTF1', 'TFF3', 'SYTL2', 'AKT1', 'IGF1R', 'ALCAM','STC2'], ncols=4, gene_symbols='feature_name')


# In[92]:


sc.tl.rank_genes_groups(NN2023_mature_luminal_hv, 'leiden')

sc.pl.rank_genes_groups(NN2023_mature_luminal_hv, n_genes=20, sharey=False)


# In[25]:


save_top_marker_genes(NN2023_basal_hv, 'NN2023', 'mature_luminal')


# # JV2021 Analysed Alone

# In[34]:


JV2021_only = sc.concat([globals()[d] for d in dataobjects_JV2021])


# In[35]:


print(JV2021_only.shape)
sc.pp.filter_genes(JV2021_only, min_cells = 200)
JV2021_only.X = csr_matrix(JV2021_only.X)
print(JV2021_only.shape)


# In[36]:


JV2021_only.var["mt"] = JV2021_only.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(
    JV2021_only, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)


# In[37]:


sc.pl.scatter(JV2021_only, x="total_counts", y="pct_counts_mt")
sc.pl.scatter(JV2021_only, x="total_counts", y="n_genes_by_counts")


# In[38]:


JV2021_only = JV2021_only[JV2021_only.obs.n_genes_by_counts < 2500, :]
JV2021_only = JV2021_only[JV2021_only.obs.pct_counts_mt < 6, :].copy()


# In[ ]:


JV2021_only.obs.groupby('Sample').count()


# In[40]:


JV2021_only.layers['counts'] = JV2021_only.X.copy()
sc.pp.normalize_total(JV2021_only, target_sum = 1e4)
sc.pp.log1p(JV2021_only)
JV2021_only.raw = JV2021_only
JV2021_only.obs.head()


# In[41]:


sc.pp.highly_variable_genes(JV2021_only, n_top_genes = 2000)
JV2021_only_hv = JV2021_only[:, JV2021_only.var['highly_variable']].copy()
sc.pp.pca(JV2021_only_hv)
sc.pp.neighbors(JV2021_only_hv, n_pcs = 20)


# In[122]:


sc.tl.leiden(JV2021_only_hv, resolution = 1.0)
sc.tl.umap(JV2021_only_hv)
sc.pl.umap(JV2021_only_hv, color=['leiden', 'Batch', 'Sample'])


# In[130]:


sc.tl.leiden(JV2021_only_hv, resolution = 1.0)
sc.tl.umap(JV2021_only_hv)
sc.pl.umap(JV2021_only_hv, color=['leiden'])


# In[133]:


sc.tl.leiden(JV2021_only_hv, resolution = 0.8)
sc.tl.umap(JV2021_only_hv)
sc.pl.umap(JV2021_only_hv, color=['leiden'])


# In[134]:


sc.tl.leiden(JV2021_only_hv, resolution = 0.5)
sc.tl.umap(JV2021_only_hv)
sc.pl.umap(JV2021_only_hv, color=['leiden'])


# ### Annotation

# In[43]:


#BASAL Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(JV2021_only_hv, color=['ACTA2', 'TAGLN', 'MYL9', 'TPM2', 'ACTG2','KRT5', 'ITGA6', 'KRT14', 'KRT17', 'CCND2', 'DKK3', 'SPARC', 'KRT14'], ncols=4)


# In[125]:


sc.pl.umap(JV2021_only_hv, color=['ACTA2', 'KRT14', 'DKK3', 'KRT14'], ncols=5)


# In[44]:


#LUMINAL PROGENITOR Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(JV2021_only_hv, color=['LTF', 'SLPI', 'RARRES1', 'ALDH1A3', 'KRT15', 'PI3', 'S100A9', 'CD24', 'KIT'], ncols=4)


# In[127]:


sc.pl.umap(JV2021_only_hv, color=['CD24'])


# In[128]:


sc.pl.umap(JV2021_only_hv, color=['LTF', 'SLPI', 'RARRES1', 'KRT15' ], ncols=4)


# In[45]:


#MATURE LUMINAL Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(JV2021_only_hv, color=['AREG', 'STC2','PRLR', 'PGR', 'ESR1', 'FOXA1', 'TTF1', 'TFF3', 'SYTL2', 'AKT1', 'IGF1R', 'ALCAM','STC2'], ncols=4)


# In[129]:


sc.pl.umap(JV2021_only_hv, color=['AREG', 'STC2','TFF3', 'SYTL2'], ncols=4)


# In[46]:


sc.pl.umap(JV2021_only_hv, color=['HLA-DRA', 'HLA-DMA', 'HLA-DRB1'], ncols=3)
#Ali: Immune (dendritic)#


# In[143]:


sc.pl.umap(JV2021_only_hv, color = ['leiden'], frameon = False, legend_loc = "on data")


# In[144]:


cell_type_JV2021 = {"0":"Luminal Mature",
"1":"Luminal Progenitor",
"2":"Basal",
"3":"Fibroblast",
"4":"Basal",
"5":"Luminal Progenitor",
"6":"Fibroblast",
"7":"Endothelial",
"8":"Endothelial",
"9":"Luminal Progenitor",
"10":"Luminal Progenitor",
"11":"Luminal Mature",
"12":"T-Cell", "13":"Other Myeloid"}


# In[142]:


marker_genes = {
    "Epithelial": ['EPCAM'],
    "Luminal":['KRT19'],
    "Basal": ['TAGLN', 'KRT14', 'ACTA2', 'KRT17', 'SAA1', 'MYLK', 'KRT5'], 
    "Luminal_Mature": ['FOXA1', 'ESR1', 'AREG', 'MUCL1', 'PIP', 'AGR2'], 
    "Luminal_Progenitor": ['KRT15', 'LTF', 'SLPI', 'EHF', 'PROM1', 'GABRP'], 
    "Adipocyte": ['APOE'], 
    "Endothelial": ['FABP5', 'MECOM', 'FABP4'], 
    "Fibroblast": ['APOD', 'TNFAIP6', 'DCN'], 
    "General_Myeloid": ['HLA-DRA', 'HLA-DPA1', 'CD74'],
    "Monocyte": ['VCAN', 'CD14'],
    "Macrophage": ['APOE', 'IDO1'], 
    "T-Cell": ['CCL5', 'CXCR4'], 
    "B-Cell": ['CD79B'] 
}

sc.pl.dotplot(
    JV2021_only_hv,
    var_names=marker_genes,
    groupby='leiden'
)


# In[146]:


JV2021_only_hv.obs['cell type'] = JV2021_only_hv.obs.leiden.map(cell_type_JV2021)


# In[147]:


sc.pl.umap(JV2021_only_hv, color = ['cell type'], frameon = False)


# In[145]:


sc.pl.umap(JV2021_only_hv, color = ['cell type'], frameon = False)


# In[148]:


JV2021_basal = JV2021_only_hv[JV2021_only_hv.obs['cell type'].isin(['Basal'])]
sc.pl.umap(JV2021_basal, color = ['cell type'], frameon = False)


# In[149]:


JV2021_mature_luminal = JV2021_only_hv[JV2021_only_hv.obs['cell type'].isin(['Luminal Mature'])]
sc.pl.umap(JV2021_mature_luminal, color = ['cell type'], frameon = False)


# In[150]:


JV2021_luminal_progenitor = JV2021_only_hv[JV2021_only_hv.obs['cell type'].isin(['Luminal Progenitor'])]
sc.pl.umap(JV2021_luminal_progenitor, color = ['cell type'], frameon = False)


# In[191]:


JV2021_epithelial = JV2021_only_hv[JV2021_only_hv.obs['cell type'].isin(['Basal', 'Luminal Mature', 'Luminal Progenitor'])]
sc.pl.umap(JV2021_epithelial, color = ['cell type'], frameon = False)


# In[152]:


JV2021_only_hv.write("./data/12_08_2025_JV2021_only_hv_annotated.h5ad")


# In[20]:


JV2021_only_hv = sc.read_h5ad("./data/12_08_2025_JV2021_only_hv_annotated.h5ad")
sc.pl.umap(JV2021_only_hv, color = ['MME', 'cell type'], frameon = False)


# In[153]:


JV2021_only_hv = sc.read_h5ad("./data/12_08_2025_JV2021_only_hv_annotated.h5ad")
print(JV2021_only_hv)
print(JV2021_only_hv.obs['cell type'].head())
sc.pl.umap(JV2021_only_hv, color=['cell type'])


# In[154]:


JV2021_only_hv.obs['cell type'].value_counts()


# ### JV2021 Markers

# In[48]:


sc.tl.rank_genes_groups(JV2021_only_hv, 'leiden')

sc.pl.rank_genes_groups(JV2021_only_hv, n_genes=20, sharey=False)


# In[139]:


markers_JV2021 = sc.get.rank_genes_groups_df(JV2021_only_hv, None)
markers_JV2021 = markers_JV2021[(markers_JV2021.pvals_adj < 0.05) & (markers_JV2021.logfoldchanges > .5)]
markers_JV2021


# In[140]:


markers_JV2021 = sc.get.rank_genes_groups_df(JV2021_only_hv, None)


# In[141]:


#BASAL#
markers_JV2021[markers_JV2021.names== 'ACTA2']


# In[142]:


#BASAL#
markers_JV2021[markers_JV2021.names== 'TAGLN']


# In[143]:


#BASAL#
markers_JV2021[markers_JV2021.names== 'TPM2']


# In[144]:


#BASAL#
markers_JV2021[markers_JV2021.names== 'KRT14']


# In[145]:


#LUMINAL PROGENITOR#
markers_JV2021[markers_JV2021.names== 'SLPI']


# In[146]:


#LUMINAL PROGENITOR#
markers_JV2021[markers_JV2021.names== 'RARRES1']


# In[147]:


#LUMINAL PROGENITOR#
markers_JV2021[markers_JV2021.names== 'S100A9']


# In[148]:


#LUMINAL PROGENITOR#
markers_JV2021[markers_JV2021.names== 'ALDH1A3']


# In[149]:


#LUMINAL MATURE#
markers_JV2021[markers_JV2021.names== 'AREG']


# In[150]:


#LUMINAL MATURE#
markers_JV2021[markers_JV2021.names== 'TFF3']


# In[151]:


#LUMINAL MATURE#
markers_JV2021[markers_JV2021.names== 'SYTL2']


# ### JV2021 Basal

# In[ ]:


JV2021_basal_hv = sc.read_h5ad("./data/17_10_2025_JV2021_basal_hv_annotated.h5ad")


# In[54]:


#sc.pp.highly_variable_genes(JV2021_basal, n_top_genes = 2000)
#JV2021_basal_hv = JV2021_basal[:, JV2021_basal.var['highly_variable']].copy()
sc.pp.pca(JV2021_basal_hv)
sc.pp.neighbors(JV2021_basal_hv, n_pcs = 20)
sc.tl.leiden(JV2021_basal_hv, resolution = 0.1)
sc.tl.umap(JV2021_basal_hv)


# In[55]:


sc.pl.umap(JV2021_basal_hv, color=['ACTA2', 'TAGLN', 'MYL9', 'TPM2', 'ACTG2', 'KRT5', 'ITGA6', 'KRT14', 'KRT17', 'CCND2', 'DKK3', 'SPARC'], ncols=4)


# In[56]:


sc.pl.umap(JV2021_basal_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[57]:


basal_leiden_JV2021 = {"0":"0", "1":"1", "2":"2"}
JV2021_basal_hv.obs['subcluster'] = JV2021_basal_hv.obs.leiden.map(basal_leiden_JV2021)

JV2021_basal_hv.write("./data/17_10_2025_JV2021_basal_hv_annotated.h5ad")


# In[67]:


NN2023_mature_luminal_hv = sc.read_h5ad("./data/17_10_2025_JV2021_basal_hv_annotated.h5ad")
NN2023_mature_luminal_hv.obs['subcluster'].value_counts()


# In[65]:


save_top_marker_genes(JV2021_basal_hv, 'JV2021', 'basal')


# In[66]:


sc.tl.rank_genes_groups(JV2021_basal_hv, 'leiden')

sc.pl.rank_genes_groups(JV2021_basal_hv, n_genes=20, sharey=False)


# ### JV2021 Luminal Progenitor

# In[130]:


JV2021_luminal_progenitor_hv = sc.read_h5ad("./data/17_10_2025_JV2021_luminal_progenitor_hv_annotated.h5ad")


# In[158]:


sc.pp.highly_variable_genes(JV2021_luminal_progenitor, n_top_genes = 2000)
JV2021_luminal_progenitor_hv = JV2021_luminal_progenitor[:, JV2021_luminal_progenitor.var['highly_variable']].copy()
sc.pp.pca(JV2021_luminal_progenitor_hv)
sc.pp.neighbors(JV2021_luminal_progenitor_hv, n_pcs = 20)
sc.tl.leiden(JV2021_luminal_progenitor_hv, resolution = 0.2)
sc.tl.umap(JV2021_luminal_progenitor_hv)


# In[159]:


#LUMINAL PROGENITOR Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(JV2021_luminal_progenitor_hv, color=['LTF', 'SLPI', 'RARRES1', 'ALDH1A3', 'KRT15', 'PI3', 'S100A9', 'CD24', 'KIT'], ncols=4)


# In[160]:


sc.pl.umap(JV2021_luminal_progenitor_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[161]:


lp_leiden_JV2021 = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4"}
JV2021_luminal_progenitor_hv.obs['subcluster'] = JV2021_luminal_progenitor_hv.obs.leiden.map(lp_leiden_JV2021)

JV2021_luminal_progenitor_hv.write("./data/17_10_2025_JV2021_luminal_progenitor_hv_annotated.h5ad")


# In[162]:


NN2023_mature_luminal_hv = sc.read_h5ad("./data/17_10_2025_JV2021_luminal_progenitor_hv_annotated.h5ad")
NN2023_mature_luminal_hv.obs['subcluster'].value_counts()


# In[164]:


save_top_marker_genes(JV2021_luminal_progenitor_hv, 'JV2021', 'luminal_progenitor')


# In[165]:


sc.tl.rank_genes_groups(JV2021_luminal_progenitor_hv, 'leiden')

sc.pl.rank_genes_groups(JV2021_luminal_progenitor_hv, n_genes=20, sharey=False)


# ### JV 2021 Luminal Mature

# In[121]:


JV2021_mature_luminal_hv = sc.read_h5ad("./data/17_10_2025_JV2021_mature_luminal_hv_annotated.h5ad")


# In[166]:


sc.pp.highly_variable_genes(JV2021_mature_luminal, n_top_genes = 2000)
JV2021_mature_luminal_hv = JV2021_mature_luminal[:, JV2021_mature_luminal.var['highly_variable']].copy()
sc.pp.pca(JV2021_mature_luminal_hv)
sc.pp.neighbors(JV2021_mature_luminal_hv, n_pcs = 20)
sc.tl.leiden(JV2021_mature_luminal_hv, resolution = 0.18)
sc.tl.umap(JV2021_mature_luminal_hv)


# In[167]:


sc.pl.umap(JV2021_mature_luminal_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[168]:


ml_leiden_JV2021 = {"0":"0", "1":"1", "2":"2"}
JV2021_mature_luminal_hv.obs['subcluster'] = JV2021_mature_luminal_hv.obs.leiden.map(ml_leiden_JV2021)

JV2021_mature_luminal_hv.write("./data/17_10_2025_JV2021_mature_luminal_hv_annotated.h5ad")


# In[169]:


NN2023_mature_luminal_hv = sc.read_h5ad("./data/17_10_2025_JV2021_mature_luminal_hv_annotated.h5ad")
NN2023_mature_luminal_hv.obs['subcluster'].value_counts()


# In[170]:


#MATURE LUMINAL Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(JV2021_mature_luminal_hv, color=['AREG', 'PRLR', 'PGR', 'ESR1', 'FOXA1', 'TTF1', 'TFF3', 'SYTL2', 'AKT1', 'IGF1R', 'ALCAM','STC2'], ncols=4)


# In[171]:


sc.tl.rank_genes_groups(JV2021_mature_luminal_hv, 'leiden')

sc.pl.rank_genes_groups(JV2021_mature_luminal_hv, n_genes=20, sharey=False)


# In[172]:


save_top_marker_genes(JV2021_mature_luminal_hv, 'JV2021', 'mature_luminal')


# # R2024 Analysed Alone

# In[94]:


R2024_only = sc.concat([globals()[d] for d in dataobjects_R2024])
#R2024_only.obs_names_make_unique()
#R2024_only.obs["Mutation_Status"] = R2024_only.obs["Sample"].map(mutation_dict).fillna("N/A")


# In[95]:


print(R2024_only.shape)
sc.pp.filter_genes(R2024_only, min_cells = 200)
R2024_only.X = csr_matrix(R2024_only.X)
print(R2024_only.shape)


# In[96]:


R2024_only.obs.groupby('Sample').count()


# In[97]:


R2024_only.var["mt"] = R2024_only.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(
    R2024_only, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)


# In[98]:


sc.pl.scatter(R2024_only, x="total_counts", y="pct_counts_mt")
sc.pl.scatter(R2024_only, x="total_counts", y="n_genes_by_counts")


# In[99]:


R2024_only = R2024_only[R2024_only.obs.n_genes_by_counts < 2500, :]
R2024_only = R2024_only[R2024_only.obs.pct_counts_mt < 6, :].copy()


# In[100]:


R2024_only.layers['counts'] = R2024_only.X.copy()
sc.pp.normalize_total(R2024_only, target_sum = 1e4)
sc.pp.log1p(R2024_only)
R2024_only.raw = R2024_only
R2024_only.obs.head()


# In[101]:


sc.pp.highly_variable_genes(R2024_only, n_top_genes = 2000)
R2024_only_hv = R2024_only[:, R2024_only.var['highly_variable']].copy()
sc.pp.pca(R2024_only_hv, n_comps=50)
sc.pp.neighbors(R2024_only_hv, n_pcs = 20)
sc.tl.leiden(R2024_only_hv, resolution = 0.9)
sc.tl.umap(R2024_only_hv)
sc.pl.umap(R2024_only_hv, color=['leiden', 'Batch', 'Sample'])


# In[105]:


#sc.pl.umap(R2024_only_hv, color=['leiden', 'Batch', 'Sample'])


# In[106]:


#BASAL Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(R2024_only_hv, color=['ACTA2', 'TAGLN', 'MYL9', 'TPM2', 'ACTG2', 'KRT5', 'ITGA6', 'KRT14', 'KRT17', 'CCND2', 'DKK3', 'SPARC', 'KRT14', 'TP63','NFIB','ACTG2','MYLK','SAA1','DST','LAMB3'], ncols=4)


# In[107]:


#LUMINAL PROGENITOR Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(R2024_only_hv, color=['LTF', 'SLPI', 'RARRES1', 'ALDH1A3', 'KRT15', 'PI3', 'S100A9', 'CD24', 'KIT', 'SFRP1','LTF','SNORC'], ncols=4)


# In[108]:


#MATURE LUMINAL Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(R2024_only_hv, color=['AREG', 'PRLR', 'PGR', 'ESR1', 'FOXA1', 'TTF1', 'TFF3', 'SYTL2', 'AKT1', 'IGF1R', 'ALCAM','STC2', 'GATA3', 'XBP1','TCIM'], ncols=4)


# In[110]:


sc.pl.umap(R2024_only_hv, color = ['leiden'], frameon = False, legend_loc = "on data")


# In[114]:


cell_type_R2024 = {"0":"Smooth muscle",
"1":"Endothelial",
"2":"Fibroblast",
"3":"Basal",
"4":"Endothelial",
"5":"Fibroblast",
"6":"Endothelial",
"7":"Basal",
                   "8":"Fibroblast",
"9":"Smooth Muscle",
"10":"Luminal Mature",
"11":"Endothelial",
"12":"Luminal Progenitor",
"13":"Basal", "14":"Basal", "15":"Immune"}


# In[115]:


R2024_only_hv.obs['cell type'] = R2024_only_hv.obs.leiden.map(cell_type_R2024)


# In[80]:


sc.pl.umap(R2024_only_hv, color = ['cell type'], ncols=1, frameon = False)


# In[79]:


R2024_basal = R2024_only_hv[R2024_only_hv.obs['cell type'].isin(['Basal'])]
sc.pl.umap(R2024_basal, color = ['cell type'],ncols=1, frameon = False)


# In[81]:


R2024_mature_luminal = R2024_only_hv[R2024_only_hv.obs['cell type'].isin(['Luminal Mature'])]
sc.pl.umap(R2024_mature_luminal, color = ['cell type'],ncols=1, frameon = False)


# In[82]:


R2024_luminal_progenitor = R2024_only_hv[R2024_only_hv.obs['cell type'].isin(['Luminal Progenitor'])]
sc.pl.umap(R2024_luminal_progenitor, color = ['cell type'],ncols=1, frameon = False)


# In[192]:


R2024_epithelial = R2024_only_hv[R2024_only_hv.obs['cell type'].isin(['Basal', 'Luminal Mature', 'Luminal Progenitor'])]
sc.pl.umap(R2024_epithelial, color = ['cell type'], ncols=1, frameon = False)


# In[84]:


R2024_only_hv.write("./data/12_08_2025_R2024_only_hv_annotated.h5ad")


# In[21]:


R2024_only_hv = sc.read_h5ad("./data/12_08_2025_R2024_only_hv_annotated.h5ad")
sc.pl.umap(R2024_only_hv, color = ['MME', 'cell type'], frameon = False)


# In[78]:


R2024_only_hv = sc.read_h5ad("./data/12_08_2025_R2024_only_hv_annotated.h5ad")
print(R2024_only_hv)
print(R2024_only_hv.obs['cell type'].head())
sc.pl.umap(R2024_only_hv, color=['cell type'])


# In[72]:


marker_genes = {
   # "Epithelial": ['EPCAM'],
    "Luminal":['KRT19'],
    "Basal": ['TAGLN', 'KRT14', 'ACTA2', 'KRT17', 'SAA1', 'MYLK', 'KRT5', 'TP63'], 
    "Luminal_Mature": ['FOXA1', 'ESR1', 'AREG'], 
    "Luminal_Progenitor": ['KRT15', 'LTF', 'SLPI'], 
    "Adipocyte": ['APOE'], 
    "Endothelial": ['FABP5', 'MECOM', 'FABP4'], 
    "Fibroblast": ['APOD', 'COL1A1', 'TNFAIP6', 'DCN', 'COL1A2'], 
    "General_Myeloid": ['HLA-DRA', 'HLA-DPA1', 'CD74'],
    "Monocyte": ['VCAN', 'CD14'],
    "Macrophage": ['APOE', 'CCL3', 'CCL4', 'IDO1'], 
    "T-Cell": ['CCL5', 'CXCR4', 'IL7R']
}

sc.pl.dotplot(
    R2024_only_hv,
    var_names=marker_genes,
    groupby='leiden'
)


# In[73]:


#old
sc.pl.umap(R2024_only_hv, color = ['cell type'], ncols=1, frameon = False)


# In[76]:


#new
sc.pl.umap(R2024_only_hv, color = ['cell type'], ncols=1, frameon = False)


# In[75]:


R2024_only_hv.obs['cell type'] = R2024_only_hv.obs.leiden.map(cell_type_R2024)


# In[74]:


cell_type_R2024 = {"0":"Smooth Muscle",
"1":"Endothelial",
"2":"Fibroblast",
"3":"Basal",
"4":"Endothelial",
"5":"Fibroblast",
"6":"Endothelial",
"7":"Smooth Muscle",
                   "8":"Fibroblast",
"9":"T-Cell",
"10":"Luminal Mature",
"11":"Endothelial",
"12":"Luminal Progenitor",
"13":"Basal", "14":"Basal", "15":"Macrophage"}


# In[120]:


R2024_only_hv.obs['cell type'].value_counts()


# ### Markers R2024

# In[111]:


sc.tl.rank_genes_groups(R2024_only_hv, 'leiden')


# In[112]:


sc.pl.rank_genes_groups(R2024_only_hv, n_genes=20, sharey=False)


# In[58]:


markers_R2024 = sc.get.rank_genes_groups_df(R2024_only_hv, None)
markers_R2024 = markers_R2024[(markers_R2024.pvals_adj < 0.05) & (markers_R2024.logfoldchanges > .5)]
markers_R2024


# In[59]:


markers_R2024 = sc.get.rank_genes_groups_df(R2024_only_hv, None)


# In[60]:


#BASAL#
markers_R2024[markers_R2024.names== 'ACTA2']


# In[61]:


#BASAL#
markers_R2024[markers_R2024.names== 'TAGLN']


# In[62]:


#BASAL#
markers_R2024[markers_R2024.names== 'TPM2']


# In[63]:


#BASAL#
markers_R2024[markers_R2024.names== 'KRT14']


# In[64]:


#LUMINAL PROGENITOR#
markers_R2024[markers_R2024.names== 'SLPI']


# In[65]:


#LUMINAL PROGENITOR#
markers_R2024[markers_R2024.names== 'RARRES1']


# In[66]:


#LUMINAL PROGENITOR#
markers_R2024[markers_R2024.names== 'ALDH1A3']


# In[67]:


#LUMINAL PROGENITOR#
markers_R2024[markers_R2024.names== 'KRT15']


# In[68]:


#LUMINAL MATURE#
markers_R2024[markers_R2024.names== 'AREG']


# In[69]:


#LUMINAL MATURE#
markers_R2024[markers_R2024.names== 'TFF3']


# In[70]:


#LUMINAL MATURE#
markers_R2024[markers_R2024.names== 'SYTL2']


# In[71]:


#LUMINAL MATURE#
markers_R2024[markers_R2024.names== 'STC2']


# In[72]:


#LUMINAL MATURE#
markers_R2024[markers_R2024.names== 'FOXA1']


# ### Basal R2024

# In[124]:


R2024_basal_hv = sc.read_h5ad("./data/17_10_2025_R2024_basal_hv_annotated.h5ad")


# In[87]:


sc.pp.highly_variable_genes(R2024_basal, n_top_genes = 2000)
R2024_basal_hv = R2024_basal[:, R2024_basal.var['highly_variable']].copy()
sc.pp.pca(R2024_basal_hv)


# In[91]:


sc.pp.neighbors(R2024_basal_hv, n_pcs = 20)
sc.tl.leiden(R2024_basal_hv, resolution = 0.2)
sc.tl.umap(R2024_basal_hv)
#eldar uses resolution 0.5#


# In[92]:


sc.pl.umap(R2024_basal_hv, color=['ACTA2', 'TAGLN', 'MYL9', 'TPM2', 'ACTG2', 'KRT5', 'ITGA6', 'KRT14', 'KRT17', 'CCND2', 'DKK3', 'SPARC'], ncols=4)


# In[93]:


sc.pl.umap(R2024_basal_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[94]:


basal_leiden_R2024 = {"0":"0", "1":"1", "2":"2", "3":"3"}
R2024_basal_hv.obs['subcluster'] = R2024_basal_hv.obs.leiden.map(basal_leiden_R2024)

R2024_basal_hv.write("./data/17_10_2025_R2024_basal_hv_annotated.h5ad")


# In[106]:


KN23_basal = sc.read_h5ad("./data/17_10_2025_R2024_basal_hv_annotated.h5ad")
KN23_basal.obs['subcluster'].value_counts()


# In[96]:


save_top_marker_genes(R2024_basal_hv, 'R2024', 'basal')


# In[163]:


sc.tl.rank_genes_groups(R2024_basal_hv, 'leiden')
sc.pl.rank_genes_groups(R2024_basal_hv, n_genes=20, sharey=False)


# ### Luminal Progenitor R2024

# In[132]:


R2024_luminal_progenitor_hv = sc.read_h5ad("./data/17_10_2025_R2024_luminal_progenitor_hv_annotated.h5ad")


# In[133]:


#sc.pp.highly_variable_genes(R2024_luminal_progenitor, n_top_genes = 2000)
#R2024_luminal_progenitor_hv = R2024_luminal_progenitor[:, R2024_luminal_progenitor.var['highly_variable']].copy()
sc.pp.pca(R2024_luminal_progenitor_hv)
sc.pp.neighbors(R2024_luminal_progenitor_hv, n_pcs = 20)
sc.tl.leiden(R2024_luminal_progenitor_hv, resolution = 0.15)
sc.tl.umap(R2024_luminal_progenitor_hv)
sc.pl.umap(R2024_luminal_progenitor_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[134]:


lp_leiden_R2024 = {"0":"0", "1":"1", "2":"2", "3":"3"}
R2024_luminal_progenitor_hv.obs['subcluster'] = R2024_luminal_progenitor_hv.obs.leiden.map(lp_leiden_R2024)

R2024_luminal_progenitor_hv.write("./data/17_10_2025_R2024_luminal_progenitor_hv_annotated.h5ad")


# In[107]:


KN23_basal = sc.read_h5ad("./data/17_10_2025_R2024_luminal_progenitor_hv_annotated.h5ad")
KN23_basal.obs['subcluster'].value_counts()


# In[135]:


#LUMINAL PROGENITOR Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(R2024_luminal_progenitor_hv, color=['LTF', 'SLPI', 'RARRES1', 'ALDH1A3', 'KRT15', 'PI3', 'S100A9', 'CD24', 'KIT'], ncols=4)


# In[136]:


sc.tl.rank_genes_groups(R2024_luminal_progenitor_hv, 'leiden')
sc.pl.rank_genes_groups(R2024_luminal_progenitor_hv, n_genes=20, sharey=False)


# In[137]:


save_top_marker_genes(R2024_luminal_progenitor_hv, 'R2024', 'luminal_progenitor')


# ### Luminal Mature R2024

# In[138]:


R2024_mature_luminal_hv = sc.read_h5ad("./data/17_10_2025_R2024_mature_luminal_hv_annotated.h5ad")


# In[140]:


#sc.pp.highly_variable_genes(R2024_mature_luminal, n_top_genes = 2000)
#R2024_mature_luminal_hv = R2024_mature_luminal[:, R2024_mature_luminal.var['highly_variable']].copy()
sc.pp.pca(R2024_mature_luminal_hv)
sc.pp.neighbors(R2024_mature_luminal_hv, n_pcs = 20)
sc.tl.leiden(R2024_mature_luminal_hv, resolution = 0.15)
sc.tl.umap(R2024_mature_luminal_hv)
sc.pl.umap(R2024_mature_luminal_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[141]:


ml_leiden_R2024 = {"0":"0", "1":"1", "2":"2"}
R2024_mature_luminal_hv.obs['subcluster'] = R2024_mature_luminal_hv.obs.leiden.map(ml_leiden_R2024)

R2024_mature_luminal_hv.write("./data/17_10_2025_R2024_mature_luminal_hv_annotated.h5ad")


# In[108]:


KN23_basal = sc.read_h5ad("./data/17_10_2025_R2024_mature_luminal_hv_annotated.h5ad")
KN23_basal.obs['subcluster'].value_counts()


# In[142]:


#MATURE LUMINAL Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(R2024_mature_luminal_hv, color=['AREG', 'PRLR', 'PGR', 'ESR1', 'FOXA1', 'TTF1', 'TFF3', 'SYTL2', 'AKT1', 'IGF1R', 'ALCAM','STC2'], ncols=4)


# In[143]:


save_top_marker_genes(R2024_mature_luminal_hv, 'R2024', 'mature_luminal')


# In[144]:


sc.tl.rank_genes_groups(R2024_mature_luminal_hv, 'leiden')
sc.pl.rank_genes_groups(R2024_mature_luminal_hv, n_genes=20, sharey=False)


# # JB2022 Analysed Alone

# In[155]:


JB2022_sample_only = ["inds_JB2022_data"]
alldata_JB2022 = sc.concat([globals()[d] for d in JB2022_sample_only])


# In[156]:


JB2022_only = alldata_JB2022
#JB2022_only = alldata_KK18[alldata_KK18.obs['Batch'].isin(['JB2022'])]


# In[157]:


print(JB2022_only.shape)
sc.pp.filter_genes(JB2022_only, min_cells = 200)
JB2022_only.X = csr_matrix(JB2022_only.X)
print(JB2022_only.shape)


# In[158]:


JB2022_only.var["mt"] = JB2022_only.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(
    JB2022_only, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)

sc.pl.scatter(JB2022_only, x="total_counts", y="pct_counts_mt")
sc.pl.scatter(JB2022_only, x="total_counts", y="n_genes_by_counts")


# In[159]:


JB2022_only = JB2022_only[JB2022_only.obs.n_genes_by_counts < 2500, :]
JB2022_only = JB2022_only[JB2022_only.obs.pct_counts_mt < 6, :].copy()


# In[160]:


JB2022_only.obs.groupby('Sample').count()


# In[161]:


JB2022_only.layers['counts'] = JB2022_only.X.copy()
sc.pp.normalize_total(JB2022_only, target_sum = 1e4)
sc.pp.log1p(JB2022_only)
JB2022_only.raw = JB2022_only
JB2022_only.obs.head()


# In[162]:


sc.pp.highly_variable_genes(JB2022_only, n_top_genes = 2000)


# In[163]:


JB2022_only_hv = JB2022_only[:, JB2022_only.var['highly_variable']].copy()
sc.pp.pca(JB2022_only_hv)
sc.pp.neighbors(JB2022_only_hv, n_pcs = 20)
sc.tl.leiden(JB2022_only_hv, resolution = 0.5)
sc.tl.umap(JB2022_only_hv)


# In[164]:


sc.pl.umap(JB2022_only_hv, color=['leiden', 'Batch', 'Sample'])


# In[190]:


#sc.pl.umap(JB2022_only_hv, color=['leiden', 'Batch', 'Sample'])


# In[165]:


#BASAL Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(JB2022_only_hv, color=['ACTA2', 'TAGLN', 'MYL9', 'TPM2', 'ACTG2', 'KRT5', 'ITGA6', 'KRT14', 'KRT17', 'CCND2', 'DKK3', 'SPARC', 'KRT14'], ncols=4)


# In[166]:


#LUMINAL PROGENITOR Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(JB2022_only_hv, color=['LTF', 'SLPI', 'RARRES1', 'ALDH1A3', 'KRT15', 'PI3', 'S100A9', 'CD24', 'KIT'], ncols=4)


# In[167]:


#MATURE LUMINAL Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(JB2022_only_hv, color=['AREG', 'PRLR', 'PGR', 'ESR1', 'FOXA1', 'TTF1', 'TFF3', 'SYTL2', 'AKT1', 'IGF1R', 'ALCAM','STC2'], ncols=4)


# In[168]:


sc.pl.umap(JB2022_only_hv, color=['HLA-DRA', 'HLA-DMA', 'HLA-DRB1'], ncols=3)
#Ali: Immune (dendritic)#


# In[169]:


sc.pl.umap(JB2022_only_hv, color = ['leiden'], frameon = False, legend_loc = "on data")


# In[172]:


cell_type_JB2022 = {"0":"Luminal Mature",
"1":"T-cell",
"2":"Luminal Progenitor",
"3":"Basal",
"4":"Basal",
"5":"Basal",
"6":"Endothelial",
"7":"Luminal Mature",
"8":"Basal",
"9":"Luminal Progenitor",
"10":"Basal",
"11":"T-cell",
"12":"Endothelial",
"13":"Plasma Cells",
"14":"Luminal Mature"}


# In[173]:


JB2022_only_hv.obs['cell type'] = JB2022_only_hv.obs.leiden.map(cell_type_JB2022)


# In[174]:


sc.pl.umap(JB2022_only_hv, color = ['cell type'], frameon = False)


# In[79]:


JB2022_basal = JB2022_only_hv[JB2022_only_hv.obs['cell type'].isin(['Basal'])]
sc.pl.umap(JB2022_basal, color = ['cell type'], frameon = False)


# In[80]:


JB2022_mature_luminal = JB2022_only_hv[JB2022_only_hv.obs['cell type'].isin(['Luminal Mature'])]
sc.pl.umap(JB2022_mature_luminal, color = ['cell type'], frameon = False)


# In[81]:


JB2022_luminal_progenitor = JB2022_only_hv[JB2022_only_hv.obs['cell type'].isin(['Luminal Progenitor'])]
sc.pl.umap(JB2022_luminal_progenitor, color = ['cell type'], frameon = False)


# In[193]:


JB2022_epithelial = JB2022_only_hv[JB2022_only_hv.obs['cell type'].isin(['Basal', 'Luminal Mature', 'Luminal Progenitor'])]
sc.pl.umap(JB2022_epithelial, color = ['cell type'], frameon = False)


# In[179]:


JB2022_only_hv.write("./data/12_08_2025_JB2022_only_hv_annotated.h5ad")


# In[22]:


JB2022_only_hv = sc.read_h5ad("./data/12_08_2025_JB2022_only_hv_annotated.h5ad")
sc.pl.umap(JB2022_only_hv, color = ['MME', 'cell type'], frameon = False)


# In[118]:


JB2022_only_hv = sc.read_h5ad("./data/12_08_2025_JB2022_only_hv_annotated.h5ad")
print(JB2022_only_hv)
print(JB2022_only_hv.obs['cell type'].head())
sc.pl.umap(JB2022_only_hv, color=['cell type'])


# In[119]:


JB2022_only_hv.obs['cell type'].value_counts()


# ### Markers JB2022

# In[170]:


sc.tl.rank_genes_groups(JB2022_only_hv, 'leiden')


# In[171]:


sc.pl.rank_genes_groups(JB2022_only_hv, n_genes=20, sharey=False)


# In[218]:


markers_JB2022 = sc.get.rank_genes_groups_df(JB2022_only_hv, None)


# In[219]:


markers_JB2022 = markers_JB2022[(markers_JB2022.pvals_adj < 0.05) & (markers_JB2022.logfoldchanges > .5)]
markers_JB2022


# In[220]:


markers_JB2022 = sc.get.rank_genes_groups_df(JB2022_only_hv, None)


# In[221]:


#BASAL#
markers_JB2022[markers_JB2022.names== 'ACTA2']


# In[222]:


#BASAL#
markers_JB2022[markers_JB2022.names== 'TAGLN']


# In[223]:


#BASAL#
markers_JB2022[markers_JB2022.names== 'TPM2']


# In[224]:


#BASAL#
markers_JB2022[markers_JB2022.names== 'KRT14']


# In[225]:


#LUMINAL PROGENITOR#
markers_JB2022[markers_JB2022.names== 'SLPI']


# In[226]:


#LUMINAL PROGENITOR#
markers_JB2022[markers_JB2022.names== 'RARRES1']


# In[227]:


#LUMINAL PROGENITOR#
markers_JB2022[markers_JB2022.names== 'S100A9']


# In[228]:


#LUMINAL PROGENITOR#
markers_JB2022[markers_JB2022.names== 'ALDH1A3']


# In[229]:


#LUMINAL MATURE#
markers_JB2022[markers_JB2022.names== 'AREG']


# In[230]:


#LUMINAL MATURE#
markers_JB2022[markers_JB2022.names== 'TFF3']


# In[231]:


#LUMINAL MATURE#
markers_JB2022[markers_JB2022.names== 'SYTL2']


# In[232]:


#LUMINAL MATURE#
markers_JB2022[markers_JB2022.names== 'STC2']


# In[233]:


#LUMINAL MATURE#
markers_JB2022[markers_JB2022.names== 'FOXA1']


# ### Basal JB2022

# In[83]:


sc.pp.highly_variable_genes(JB2022_basal, n_top_genes = 2000)
JB2022_basal_hv = JB2022_basal[:, JB2022_basal.var['highly_variable']].copy()
sc.pp.pca(JB2022_basal_hv)
sc.pp.neighbors(JB2022_basal_hv, n_pcs = 20)
sc.tl.leiden(JB2022_basal_hv, resolution = 0.2)
sc.tl.umap(JB2022_basal_hv)


# In[181]:


sc.pl.umap(JB2022_basal_hv, color=['ACTA2', 'TAGLN', 'MYL9', 'TPM2', 'ACTG2', 'KRT5', 'ITGA6', 'KRT14', 'KRT17', 'CCND2', 'DKK3', 'SPARC'], ncols=4)


# In[84]:


sc.pl.umap(JB2022_basal_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[85]:


basal_leiden_JB2022 = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4", "5":"5"}
JB2022_basal_hv.obs['subcluster'] = JB2022_basal_hv.obs.leiden.map(basal_leiden_JB2022)

JB2022_basal_hv.write("./data/17_10_2025_JB2022_basal_hv_annotated.h5ad")


# In[109]:


KN23_basal = sc.read_h5ad("./data/17_10_2025_JB2022_basal_hv_annotated.h5ad")
KN23_basal.obs['subcluster'].value_counts()


# In[206]:


sc.tl.rank_genes_groups(JB2022_basal_hv, 'leiden')


# In[183]:


save_top_marker_genes(JB2022_basal_hv, 'JB2022', 'basal')


# In[252]:


sc.pl.rank_genes_groups(JB2022_basal_hv, n_genes=20, sharey=False)


# ### Luminal Progenitor JB2022

# In[86]:


sc.pp.highly_variable_genes(JB2022_luminal_progenitor, n_top_genes = 2000)
JB2022_luminal_progenitor_hv = JB2022_luminal_progenitor[:, JB2022_luminal_progenitor.var['highly_variable']].copy()
sc.pp.pca(JB2022_luminal_progenitor_hv)
sc.pp.neighbors(JB2022_luminal_progenitor_hv, n_pcs = 20)
sc.tl.leiden(JB2022_luminal_progenitor_hv, resolution = 0.1)
sc.tl.umap(JB2022_luminal_progenitor_hv)


# In[185]:


#LUMINAL PROGENITOR Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(JB2022_luminal_progenitor_hv, color=['LTF', 'SLPI', 'RARRES1', 'ALDH1A3', 'KRT15', 'PI3', 'S100A9', 'CD24', 'KIT'], ncols=4)


# In[87]:


sc.pl.umap(JB2022_luminal_progenitor_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[88]:


lp_leiden_JB2022 = {"0":"0", "1":"1", "2":"2"}
JB2022_luminal_progenitor_hv.obs['subcluster'] = JB2022_luminal_progenitor_hv.obs.leiden.map(lp_leiden_JB2022)

JB2022_luminal_progenitor_hv.write("./data/17_10_2025_JB2022_luminal_progenitor_hv_annotated.h5ad")


# In[110]:


KN23_basal = sc.read_h5ad("./data/17_10_2025_JB2022_luminal_progenitor_hv_annotated.h5ad")
KN23_basal.obs['subcluster'].value_counts()


# In[189]:


save_top_marker_genes(JB2022_luminal_progenitor_hv, 'JB2022', 'luminal_progenitor')


# In[ ]:


sc.tl.rank_genes_groups(JB2022_luminal_progenitor_hv, 'leiden')


# In[ ]:


sc.pl.rank_genes_groups(JB2022_luminal_progenitor_hv, n_genes=20, sharey=False)


# ### Mature Luminal JB2022

# In[89]:


sc.pp.highly_variable_genes(JB2022_mature_luminal, n_top_genes = 2000)
JB2022_mature_luminal_hv = JB2022_mature_luminal[:, JB2022_mature_luminal.var['highly_variable']].copy()
sc.pp.pca(JB2022_mature_luminal_hv)
sc.pp.neighbors(JB2022_mature_luminal_hv, n_pcs = 20)
sc.tl.leiden(JB2022_mature_luminal_hv, resolution = 0.3)
sc.tl.umap(JB2022_mature_luminal_hv)


# In[90]:


sc.pl.umap(JB2022_mature_luminal_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[92]:


ml_leiden_JB2022 = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4"}
JB2022_mature_luminal_hv.obs['subcluster'] = JB2022_mature_luminal_hv.obs.leiden.map(ml_leiden_JB2022)

JB2022_mature_luminal_hv.write("./data/17_10_2025_JB2022_mature_luminal_hv_annotated.h5ad")


# In[111]:


KN23_basal = sc.read_h5ad("./data/17_10_2025_JB2022_mature_luminal_hv_annotated.h5ad")
KN23_basal.obs['subcluster'].value_counts()


# In[192]:


#MATURE LUMINAL Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(JB2022_mature_luminal_hv, color=['AREG', 'PRLR', 'PGR', 'ESR1', 'FOXA1', 'TTF1', 'TFF3', 'SYTL2', 'AKT1', 'IGF1R', 'ALCAM','STC2'], ncols=4)


# In[215]:


sc.tl.rank_genes_groups(JB2022_mature_luminal_hv, 'leiden')


# In[193]:


save_top_marker_genes(JB2022_mature_luminal_hv, 'JB2022', 'mature_luminal')


# In[ ]:


sc.pl.rank_genes_groups(JB2022_mature_luminal_hv, n_genes=20, sharey=False)


# # KK18 Analysed Alone

# In[194]:


#KK18_only = alldata_KK18[alldata_KK18.obs['Batch'].isin(['KK18'])]
KK18_only = sc.concat([globals()[d] for d in dataobjects_KK18])


# In[195]:


print(KK18_only.shape)
sc.pp.filter_genes(KK18_only, min_cells = 200)
KK18_only.X = csr_matrix(KK18_only.X)
print(KK18_only.shape)


# In[196]:


KK18_only.var["mt"] = KK18_only.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(
    KK18_only, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)


# In[197]:


sc.pl.scatter(KK18_only, x="total_counts", y="pct_counts_mt")
sc.pl.scatter(KK18_only, x="total_counts", y="n_genes_by_counts")


# In[198]:


KK18_only = KK18_only[KK18_only.obs.n_genes_by_counts < 2500, :]
KK18_only = KK18_only[KK18_only.obs.pct_counts_mt < 6, :].copy()


# In[199]:


KK18_only.obs.groupby('Sample').count()


# In[200]:


KK18_only.layers['counts'] = KK18_only.X.copy()
sc.pp.normalize_total(KK18_only, target_sum = 1e4)
sc.pp.log1p(KK18_only)
KK18_only.raw = KK18_only
KK18_only.obs.head()


# In[201]:


sc.pp.highly_variable_genes(KK18_only, n_top_genes = 2000)
KK18_only_hv = KK18_only[:, KK18_only.var['highly_variable']].copy()
sc.pp.pca(KK18_only_hv)


# In[211]:


sc.pp.neighbors(KK18_only_hv, n_pcs = 20)
sc.tl.leiden(KK18_only_hv, resolution = 0.5)
sc.tl.umap(KK18_only_hv)


# In[212]:


sc.pl.umap(KK18_only_hv, color=['leiden', 'Batch', 'Sample'])


# In[213]:


#BASAL Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(KK18_only_hv, color=['ACTA2', 'TAGLN', 'MYL9', 'TPM2', 'ACTG2', 'KRT5', 'ITGA6', 'KRT14', 'KRT17', 'CCND2', 'DKK3', 'SPARC', 'KRT14'], ncols=4)


# In[214]:


#BASAL Markers for Epithelial Breast Subtypes:Human Protein Atlast 'Myoepithelial Breast#
sc.pl.umap(KK18_only_hv, color=['DKK3', 'LAMC2', 'SERPINB5'], ncols=4) 


# In[215]:


#LUMINAL PROGENITOR Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(KK18_only_hv, color=['LTF', 'SLPI', 'RARRES1', 'ALDH1A3', 'KRT15', 'PI3', 'S100A9', 'CD24', 'KIT'], ncols=4)


# In[216]:


#MATURE LUMINAL Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(KK18_only_hv, color=['AREG', 'PRLR', 'PGR', 'ESR1', 'FOXA1', 'TTF1', 'TFF3', 'SYTL2', 'AKT1', 'IGF1R', 'ALCAM','STC2'], ncols=4)


# In[217]:


sc.pl.umap(KK18_only_hv, color=['HLA-DRA', 'HLA-DMA', 'HLA-DRB1'], ncols=3)
#Ali: Immune (dendritic)#


# In[218]:


sc.pl.umap(KK18_only_hv, color = ['leiden'], frameon = False, legend_loc = "on data")


# In[220]:


cell_type_KK18 = {"0":"Basal",
"1":"Basal",
"2":"Luminal Progenitor",
"3":"Basal",
"4":"Luminal Mature",
"5":"Luminal Progenitor",
"6":"Luminal Progenitor",
"7":"Luminal Mature",
"8":"Luminal Progenitor",
"9":"Basal",
"10":"Basal",
"11":"Basal"}


# In[221]:


KK18_only_hv.obs['cell type'] = KK18_only_hv.obs.leiden.map(cell_type_KK18)


# In[222]:


sc.pl.umap(KK18_only_hv, color = ['cell type'], frameon = False)


# In[94]:


KK18_basal = KK18_only_hv[KK18_only_hv.obs['cell type'].isin(['Basal'])]
sc.pl.umap(KK18_basal, color = ['cell type'], frameon = False)


# In[95]:


KK18_mature_luminal = KK18_only_hv[KK18_only_hv.obs['cell type'].isin(['Luminal Mature'])]
sc.pl.umap(KK18_mature_luminal, color = ['cell type'], frameon = False)


# In[96]:


KK18_luminal_progenitor = KK18_only_hv[KK18_only_hv.obs['cell type'].isin(['Luminal Progenitor'])]
sc.pl.umap(KK18_luminal_progenitor, color = ['cell type'], frameon = False)


# In[194]:


KK18_epithelial = KK18_only_hv[KK18_only_hv.obs['cell type'].isin(['Basal', 'Luminal Mature', 'Luminal Progenitor'])]
sc.pl.umap(KK18_epithelial, color = ['cell type'], frameon = False)


# In[227]:


KK18_only_hv.write("./data/12_08_2025_KK18_only_hv_annotated.h5ad")


# In[23]:


KK18_only_hv = sc.read_h5ad("./data/12_08_2025_KK18_only_hv_annotated.h5ad")
sc.pl.umap(KK18_only_hv, color = ['MME', 'cell type'], frameon = False)


# In[116]:


KK18_only_hv = sc.read_h5ad("./data/12_08_2025_KK18_only_hv_annotated.h5ad")
print(KK18_only_hv)
print(KK18_only_hv.obs['cell type'].head())
sc.pl.umap(KK18_only_hv, color=['cell type'])


# In[117]:


KK18_only_hv.obs['cell type'].value_counts()


# ### KK18 Markers

# In[219]:


sc.tl.rank_genes_groups(KK18_only_hv, 'leiden')

sc.pl.rank_genes_groups(KK18_only_hv, n_genes=20, sharey=False)


# In[287]:


markers_KK18 = sc.get.rank_genes_groups_df(KK18_only_hv, None)
markers_KK18 = markers_KK18[(markers_KK18.pvals_adj < 0.05) & (markers_KK18.logfoldchanges > .5)]
markers_KK18


# In[288]:


markers_KK18 = sc.get.rank_genes_groups_df(KK18_only_hv, None)


# In[289]:


#BASAL#
markers_KK18[markers_KK18.names== 'ACTA2']


# In[290]:


#BASAL#
markers_KK18[markers_KK18.names== 'TAGLN']


# In[291]:


#BASAL#
markers_KK18[markers_KK18.names== 'TPM2']


# In[292]:


#BASAL#
markers_KK18[markers_KK18.names== 'KRT14']


# In[293]:


#LUMINAL PROGENITOR#
markers_KK18[markers_KK18.names== 'SLPI']


# In[294]:


#LUMINAL PROGENITOR#
markers_KK18[markers_KK18.names== 'RARRES1']


# In[295]:


#LUMINAL PROGENITOR#
markers_KK18[markers_KK18.names== 'S100A9']


# In[296]:


#LUMINAL PROGENITOR#
markers_KK18[markers_KK18.names== 'ALDH1A3']


# In[297]:


#LUMINAL MATURE#
markers_KK18[markers_KK18.names== 'AREG']


# In[298]:


#LUMINAL MATURE#
markers_KK18[markers_KK18.names== 'TFF3']


# In[299]:


#LUMINAL MATURE#
markers_KK18[markers_KK18.names== 'SYTL2']


# In[300]:


#LUMINAL MATURE#
markers_KK18[markers_KK18.names== 'STC2']


# In[301]:


#LUMINAL MATURE#
markers_KK18[markers_KK18.names== 'FOXA1']


# ### KK18 Basal

# In[97]:


sc.pp.highly_variable_genes(KK18_basal, n_top_genes = 2000)
KK18_basal_hv = KK18_basal[:, KK18_basal.var['highly_variable']].copy()
sc.pp.pca(KK18_basal_hv)
sc.pp.neighbors(KK18_basal_hv, n_pcs = 20)
sc.tl.leiden(KK18_basal_hv, resolution = 0.3)
sc.tl.umap(KK18_basal_hv)
sc.pl.umap(KK18_basal_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[98]:


basal_leiden_KK18 = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4", "5":"5", "6":"6"}
KK18_basal_hv.obs['subcluster'] = KK18_basal_hv.obs.leiden.map(basal_leiden_KK18)

KK18_basal_hv.write("./data/17_10_2025_KK18_basal_hv_annotated.h5ad")


# In[115]:


KN23_basal = sc.read_h5ad("./data/17_10_2025_KK18_mature_luminal_hv_annotated.h5ad")
KN23_basal.obs['subcluster'].value_counts()


# In[233]:


sc.pl.umap(KK18_basal_hv, color=['ACTA2', 'TAGLN', 'MYL9', 'TPM2', 'ACTG2', 'KRT5', 'ITGA6', 'KRT14', 'KRT17', 'CCND2', 'DKK3', 'SPARC'], ncols=4)


# In[235]:


save_top_marker_genes(KK18_basal_hv, 'KK18', 'basal')


# In[246]:


sc.tl.rank_genes_groups(KK18_basal_hv, 'leiden')
sc.pl.rank_genes_groups(KK18_basal_hv, n_genes=20, sharey=False)


# ### KK18 Luminal Progenitor

# In[99]:


sc.pp.highly_variable_genes(KK18_luminal_progenitor, n_top_genes = 2000)
KK18_luminal_progenitor_hv = KK18_luminal_progenitor[:, KK18_luminal_progenitor.var['highly_variable']].copy()
sc.pp.pca(KK18_luminal_progenitor_hv)


# In[100]:


sc.pp.neighbors(KK18_luminal_progenitor_hv, n_pcs = 20)
sc.tl.leiden(KK18_luminal_progenitor_hv, resolution = 0.2)
sc.tl.umap(KK18_luminal_progenitor_hv)


# In[ ]:


basal_leiden_KK18 = {"0":"0", "1":"1", "2":"2", "3":"3", "4":"4", "5":"5", "6":"6"}
KK18_basal_hv.obs['subcluster'] = KK18_basal_hv.obs.leiden.map(basal_leiden_KK18)

KK18_basal_hv.write("./data/17_10_2025_KK18_basal_hv_annotated.h5ad")


# In[113]:


KN23_basal = sc.read_h5ad("./data/17_10_2025_KK18_basal_hv_annotated.h5ad")
KN23_basal.obs['subcluster'].value_counts()


# In[238]:


#LUMINAL PROGENITOR Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(KK18_luminal_progenitor_hv, color=['LTF', 'SLPI', 'RARRES1', 'ALDH1A3', 'KRT15', 'PI3', 'S100A9', 'CD24', 'KIT'], ncols=4)


# In[101]:


sc.pl.umap(KK18_luminal_progenitor_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[251]:


sc.tl.rank_genes_groups(KK18_luminal_progenitor_hv, 'leiden')


# In[240]:


save_top_marker_genes(KK18_luminal_progenitor_hv, 'KK18', 'luminal_progenitor')


# In[253]:


sc.pl.rank_genes_groups(KK18_luminal_progenitor_hv, n_genes=20, sharey=False)


# ### KK18 Mature Luminal

# In[241]:


sc.pp.highly_variable_genes(KK18_mature_luminal, n_top_genes = 2000)
KK18_mature_luminal_hv = KK18_mature_luminal[:, KK18_mature_luminal.var['highly_variable']].copy()
sc.pp.pca(KK18_mature_luminal_hv)


# In[245]:


sc.pp.neighbors(KK18_mature_luminal_hv, n_pcs = 20)
sc.tl.leiden(KK18_mature_luminal_hv, resolution = 0.15)
sc.tl.umap(KK18_mature_luminal_hv)


# In[246]:


sc.pl.umap(KK18_mature_luminal_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[247]:


#MATURE LUMINAL Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(KK18_mature_luminal_hv, color=['AREG', 'PRLR', 'PGR', 'ESR1', 'FOXA1', 'TTF1', 'TFF3', 'SYTL2', 'AKT1', 'IGF1R', 'ALCAM','STC2'], ncols=4)


# In[248]:


save_top_marker_genes(KK18_mature_luminal_hv, 'KK18', 'mature_luminal')


# In[167]:


sc.tl.rank_genes_groups(KK18_mature_luminal_hv, 'leiden')


# In[168]:


sc.pl.rank_genes_groups(KK18_mature_luminal_hv, n_genes=20, sharey=False)


# # All Studies' Epithelial Cell Type Markers

# In[195]:


alldatasets_epithelial = [N2021_epithelial, R2024_epithelial, KK18_epithelial, JV2021_epithelial, JB2022_epithelial, NN2023_epithelial, KN23_epithelial]

alldata_epithelial = sc.concat(alldatasets_epithelial, join='outer')  # or 'inner' depending on your needs


# In[196]:


sc.pp.highly_variable_genes(alldata_epithelial, n_top_genes = 2000)
alldata_epithelial_hv = alldata_epithelial[:, alldata_epithelial.var['highly_variable']].copy()
sc.pp.pca(alldata_epithelial_hv)
sc.pp.neighbors(alldata_epithelial_hv, n_pcs = 20)
sc.tl.leiden(alldata_epithelial_hv, resolution = 0.3)
sc.tl.umap(alldata_epithelial_hv)
sc.pl.umap(alldata_epithelial_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[199]:


sc.pl.umap(alldata_epithelial_hv, color=['Batch', 'Sample'], ncols=2)


# In[197]:


def save_top_marker_genes_by_celltype(adata, study_name, cell_types, gene_counts=[20, 30, 100]):
    """
    Saves the top differentially expressed marker genes for each cell type by comparing to other cell types in the same study.
    
    Parameters:
    - adata: AnnData object containing all epithelial cells from a study
    - study_name: str, the name of the study (e.g., 'N2021')
    - cell_types: list, cell types to analyze (e.g., ['basal', 'luminal_progenitor', 'mature_luminal'])
    - gene_counts: list, number of top genes to save (default: [20, 30, 100])
    """
    output_dir = "All_epithelial"
    os.makedirs(output_dir, exist_ok=True)
    
    for cell_type in cell_types:
        #Run differential expression analysis comparing this cell type to all others
        sc.tl.rank_genes_groups(adata, groupby='cell type', reference='rest')
        
        for n_genes in gene_counts:
            #Get top differentially expressed genes for the cell type
            top_genes_df = sc.get.rank_genes_groups_df(adata, group=cell_type)[:n_genes]
            
            #Define filename
            filename = f"{output_dir}/up{n_genes}_{study_name}_{cell_type}.csv"
            
            #Save to CSV
            top_genes_df.to_csv(filename, index=False)
            print(f"Saved: {filename}")
import os


# In[198]:


save_top_marker_genes_by_celltype(N2021_epithelial, 'N2021', ['Basal', 'Luminal Progenitor', 'Luminal Mature'])
save_top_marker_genes_by_celltype(JV2021_epithelial, 'JV2021', ['Basal', 'Luminal Progenitor', 'Luminal Mature'])
save_top_marker_genes_by_celltype(JB2022_epithelial, 'JB2022', ['Basal', 'Luminal Progenitor', 'Luminal Mature'])
save_top_marker_genes_by_celltype(R2024_epithelial, 'R2024', ['Basal', 'Luminal Progenitor', 'Luminal Mature'])
save_top_marker_genes_by_celltype(KK18_epithelial, 'KK18', ['Basal', 'Luminal Progenitor', 'Luminal Mature'])
save_top_marker_genes_by_celltype(KN23_epithelial, 'KN23', ['Basal', 'Luminal Progenitor', 'Luminal Mature'])
save_top_marker_genes_by_celltype(NN2023_epithelial, 'NN2023', ['Basal', 'Luminal Progenitor', 'Luminal Mature'])


# # Differential Gene Expression Comparison

# ### Simplified Comparison: Rank-Based Correlation ###

# In[97]:


def build_rank_dendrogram(folder_path, top_n=None, title=None, method='average', min_common_genes=5):
    """
    Build and plot a dendrogram from ranked gene signatures in CSV files.
    
    Parameters
    ----------
    folder_path : str
        Path to folder containing CSVs with columns ['names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj'].
    top_n : int or None
        Use only the top N genes per file (after filtering for upregulated). None uses all.
    title : str or None
        Title for the dendrogram plot.
    method : str
        Linkage method for hierarchical clustering.
    min_common_genes : int
        Minimum number of genes required to compute correlation between two clusters.
    """
    
    # === Step 1: Load CSVs ===
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")

    # === Step 2: Build rank signatures ===
    def build_rank_signature(df):
        df = df.copy()
        df = df[df['logfoldchanges'] > 0]  # keep only upregulated genes
        df = df.sort_values('logfoldchanges', ascending=False)
        if top_n is not None:
            df = df.head(top_n)
        df['rank'] = np.arange(1, len(df) + 1)
        return df.set_index('names')['rank']

    rank_signatures = {}
    for file_path in csv_files:
        cluster_name = os.path.basename(file_path).replace('.csv', '')
        df = pd.read_csv(file_path)
        rank_signatures[cluster_name] = build_rank_signature(df)

    # === Step 3: Compute Spearman correlation matrix ===
    cluster_names = list(rank_signatures.keys())
    S = pd.DataFrame(index=cluster_names, columns=cluster_names, dtype=float)

    for i, ki in enumerate(cluster_names):
        for j, kj in enumerate(cluster_names):
            if j < i:
                continue
            common_genes = rank_signatures[ki].index.intersection(rank_signatures[kj].index)
            if len(common_genes) < min_common_genes:
                corr_val = np.nan
            else:
                corr_val, _ = spearmanr(rank_signatures[ki].loc[common_genes],
                                        rank_signatures[kj].loc[common_genes])
            S.loc[ki, kj] = corr_val
            S.loc[kj, ki] = corr_val

    # === Step 4: Build dendrogram ===
    dist_matrix = 1 - S.astype(float)
    dist_matrix = dist_matrix.fillna(1)
    np.fill_diagonal(dist_matrix.values, 0)

    dist_vector = squareform(dist_matrix.values)
    linkage_matrix = linkage(dist_vector, method=method)

    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix, labels=cluster_names, leaf_rotation=90)
    plt.title(title or f"Cluster similarity dendrogram (top {top_n} genes)")
    plt.ylabel("1 - Spearman correlation")
    plt.tight_layout()
    plt.show()

# === Example usage ===
build_rank_dendrogram('./data/basal/up100/', top_n=100, title="Basal subcluster similarity (top 100)")


# In[98]:


build_rank_dendrogram('./data/basal/up30/', top_n=30, title="Basal subcluster similarity (top 30)")


# In[99]:


build_rank_dendrogram('./data/basal/up1000/', top_n=1000, title="Basal subcluster similarity (top 1000)")


# In[100]:


build_rank_dendrogram('./data/basal/up20/', top_n=20, title="Basal subcluster similarity (top 20)")


# In[180]:


build_rank_dendrogram('./data/mature_luminal/up1000/', top_n=1000, title="Luminal Mature subcluster similarity (top 1000)")


# In[181]:


build_rank_dendrogram('./data/mature_luminal/up100/', top_n=100, title="Luminal Mature subcluster similarity (top 100)")


# In[182]:


build_rank_dendrogram('./data/mature_luminal/up30/', top_n=30, title="Luminal Mature subcluster similarity (top 30)")


# In[183]:


build_rank_dendrogram('./data/mature_luminal/up20/', top_n=20, title="Luminal Mature subcluster similarity (top 20)")


# In[184]:


build_rank_dendrogram('./data/luminal_progenitor/up1000/', top_n=1000, title="Luminal Progenitor subcluster similarity (top 1000)")


# In[185]:


build_rank_dendrogram('./data/luminal_progenitor/up100/', top_n=100, title="Luminal Progenitor subcluster similarity (top 100)")


# In[186]:


build_rank_dendrogram('./data/luminal_progenitor/up30/', top_n=30, title="Luminal Progenitor subcluster similarity (top 30)")


# In[14]:


build_rank_dendrogram('./data/luminal_progenitor/up20/', top_n=20, title="Luminal Progenitor subcluster similarity (top 20)")


# In[200]:


build_rank_dendrogram('./data/All_epithelial/Up100/', top_n=100, title="All Epithelial subcluster similarity (top 100)")


# In[201]:


build_rank_dendrogram('./data/All_epithelial/Up30/', top_n=30, title="All Epithelial subcluster similarity (top 30)")


# In[202]:


build_rank_dendrogram('./data/All_epithelial/Up20/', top_n=20, title="All Epithelial subcluster similarity (top 20)")


# ## Evaluate Subclustering

# In[170]:


from scipy.cluster.hierarchy import fcluster

def evaluate_cuts(linkage_matrix, cluster_names, max_k=10):
    """
    Evaluate cluster stability across different cut numbers (k).
    Returns assignments for each k.
    """
    results = {}
    for k in range(2, max_k+1):
        clusters = fcluster(linkage_matrix, k, criterion='maxclust')
        assignment = pd.Series(clusters, index=cluster_names)
        results[k] = assignment
    return results


# In[171]:


def permutation_stability_test(rank_signatures, base_linkage, cluster_names,
                               min_common_genes=5, n_iter=100, max_k=10):

    # Base cut evaluations
    base_clusters = evaluate_cuts(base_linkage, cluster_names, max_k=max_k)

    stability = {k: [] for k in range(2, max_k+1)}

    for it in range(n_iter):
        permuted = {}

        # Step 1 — randomly drop 5% genes
        for name, sig in rank_signatures.items():
            keep = np.random.rand(len(sig)) > 0.05
            permuted[name] = sig[keep]

        # Step 2 — recompute similarity matrix
        S = pd.DataFrame(index=cluster_names, columns=cluster_names, dtype=float)

        for i, a in enumerate(cluster_names):
            for j, b in enumerate(cluster_names):
                if j < i:
                    continue
                common = permuted[a].index.intersection(permuted[b].index)
                if len(common) < min_common_genes:
                    corr_val = np.nan
                else:
                    corr_val, _ = spearmanr(permuted[a][common], permuted[b][common])
                S.loc[a, b] = S.loc[b, a] = corr_val

        dist = 1 - S.fillna(1).astype(float)
        np.fill_diagonal(dist.values, 0)
        dist_vector = squareform(dist.values)
        perm_linkage = linkage(dist_vector, method='average')

        # Step 3 — compare partitions at each k
        perm_assignments = evaluate_cuts(perm_linkage, cluster_names, max_k=max_k)

        for k in range(2, max_k+1):
            base = base_clusters[k]
            perm = perm_assignments[k]

            # fraction of pairs that remained in same cluster
            same_pairs = []
            for i in cluster_names:
                for j in cluster_names:
                    same_base = base[i] == base[j]
                    same_perm = perm[i] == perm[j]
                    same_pairs.append(int(same_base == same_perm))

            stability[k].append(np.mean(same_pairs))

    return pd.DataFrame({k: stability[k] for k in stability})


# In[191]:


stab = permutation_stability_test(rank_signatures, linkage_matrix, cluster_names)


# In[192]:


import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------------------
# Step 1 — Build rank signatures + linkage
# -----------------------------------------

def build_rank_signatures_and_linkage(folder_path, top_n=100, min_common_genes=5, method="average"):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in: {folder_path}")

    # Build ranked signatures
    def build_rank_signature(df):
        df = df[df["logfoldchanges"] > 0]                   # keep only upregulated
        df = df.sort_values("logfoldchanges", ascending=False)
        df = df.head(top_n)
        df["rank"] = np.arange(1, len(df) + 1)
        return df.set_index("names")["rank"]

    rank_signatures = {}
    for fp in csv_files:
        cname = os.path.basename(fp).replace(".csv", "")
        df = pd.read_csv(fp)
        rank_signatures[cname] = build_rank_signature(df)

    cluster_names = list(rank_signatures.keys())

    # Compute correlation matrix
    S = pd.DataFrame(index=cluster_names, columns=cluster_names, dtype=float)
    for i, c1 in enumerate(cluster_names):
        for j, c2 in enumerate(cluster_names):
            if j < i:
                continue
            common = rank_signatures[c1].index.intersection(rank_signatures[c2].index)
            if len(common) < min_common_genes:
                corr = np.nan
            else:
                corr, _ = spearmanr(rank_signatures[c1][common], rank_signatures[c2][common])
            S.loc[c1, c2] = corr
            S.loc[c2, c1] = corr

    # Build distance matrix
    D = 1 - S.fillna(1)
    np.fill_diagonal(D.values, 0)
    dist_vec = squareform(D.values)

    Z = linkage(dist_vec, method=method)

    return rank_signatures, Z, cluster_names, D


# -------------------------------------------------------
# Step 2 — Permutation stability test on cluster dendrogram
# -------------------------------------------------------

def permutation_stability_test(rank_signatures, base_linkage, cluster_names,
                               min_common_genes=5, n_iter=100, max_k=10):

    def cluster_labels(Z, k):
        return fcluster(Z, k, criterion="maxclust")

    base_labels = {k: cluster_labels(base_linkage, k) for k in range(2, max_k + 1)}

    results = pd.DataFrame(index=range(2, max_k + 1), columns=["mean_stability"])

    for k in range(2, max_k + 1):
        stabilities = []
        for _ in tqdm(range(n_iter), desc=f"k={k}"):
            # Permute: random 5% drop-out
            perm_sigs = {}
            for name, series in rank_signatures.items():
                mask = np.random.rand(len(series)) > 0.05
                perm_sigs[name] = series[mask]

            # Recompute similarity
            S = pd.DataFrame(index=cluster_names, columns=cluster_names, dtype=float)
            for i, c1 in enumerate(cluster_names):
                for j, c2 in enumerate(cluster_names):
                    if j < i:
                        continue
                    common = perm_sigs[c1].index.intersection(perm_sigs[c2].index)
                    if len(common) < min_common_genes:
                        corr = np.nan
                    else:
                        corr, _ = spearmanr(perm_sigs[c1][common], perm_sigs[c2][common])
                    S.loc[c1, c2] = corr
                    S.loc[c2, c1] = corr

            D = 1 - S.fillna(1)
            np.fill_diagonal(D.values, 0)
            Zp = linkage(squareform(D.values), method="average")

            perm_labels = cluster_labels(Zp, k)

            stability = (perm_labels == base_labels[k]).mean()
            stabilities.append(stability)

        results.loc[k, "mean_stability"] = np.mean(stabilities)

    return results


# In[193]:


folder = "./data/basal/up100/"

rank_sigs, Z, names, D = build_rank_signatures_and_linkage(
    folder_path=folder,
    top_n=100,
    min_common_genes=5,
    method="average"
)

stability_results = permutation_stability_test(
    rank_signatures=rank_sigs,
    base_linkage=Z,
    cluster_names=names,
    min_common_genes=5,
    n_iter=100,
    max_k=30
)

print(stability_results)


# In[195]:


from scipy.cluster.hierarchy import dendrogram

plt.figure(figsize=(10,6))
dendrogram(Z, labels=names, leaf_rotation=90)
plt.title("Basal subcluster dendrogram (top 100 markers)")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(stability_results.index, stability_results["mean_stability"], marker="o")
plt.ylabel("Cluster stability Basal Up100")
plt.xlabel("Number of clusters (k)")
plt.title("Permutation-based cluster stability (Basal Up100)")
plt.grid(True)
plt.show()


# In[ ]:





# In[187]:


import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from tqdm import tqdm

def permutation_cluster_significance(rank_signatures, method="average",
                                     min_common_genes=5, n_iter=100, max_k=10):
    """
    Compute permutation-based cluster stability/significance.

    rank_signatures: dict
        Keys are sample names, values are pd.Series of ranked genes.
    method: str
        Linkage method for hierarchical clustering.
    min_common_genes: int
        Minimum shared genes to compute Spearman correlation.
    n_iter: int
        Number of permutations.
    max_k: int
        Max number of clusters to evaluate.

    Returns
    -------
    stability_results: pd.DataFrame
        Mean stability per number of clusters (k).
    """

    # 1. Compute original dendrogram
    cluster_names = list(rank_signatures.keys())
    N = len(cluster_names)
    S = pd.DataFrame(index=cluster_names, columns=cluster_names, dtype=float)

    # Compute Spearman correlation matrix
    for i, c1 in enumerate(cluster_names):
        for j, c2 in enumerate(cluster_names):
            if j < i:
                continue
            common = rank_signatures[c1].index.intersection(rank_signatures[c2].index)
            if len(common) < min_common_genes:
                corr = np.nan
            else:
                corr, _ = spearmanr(rank_signatures[c1][common], rank_signatures[c2][common])
            S.loc[c1, c2] = corr
            S.loc[c2, c1] = corr

    D = 1 - S.fillna(1)
    np.fill_diagonal(D.values, 0)
    dist_vec = squareform(D.values)
    Z = linkage(dist_vec, method=method)

    # Function to get cluster labels
    def get_labels(Z, k):
        return fcluster(Z, k, criterion="maxclust")

    base_labels = {k: get_labels(Z, k) for k in range(2, max_k + 1)}

    # 2. Permutation test
    stability_results = pd.DataFrame(index=range(2, max_k + 1), columns=["mean_stability"])

    for k in range(2, max_k + 1):
        stabilities = []
        for _ in tqdm(range(n_iter), desc=f"k={k}"):
            # Permute by removing 5% of genes randomly per sample
            perm_signatures = {}
            for name, series in rank_signatures.items():
                mask = np.random.rand(len(series)) > 0.05
                perm_signatures[name] = series[mask]

            # Recompute correlation matrix
            S_perm = pd.DataFrame(index=cluster_names, columns=cluster_names, dtype=float)
            for i, c1 in enumerate(cluster_names):
                for j, c2 in enumerate(cluster_names):
                    if j < i:
                        continue
                    common = perm_signatures[c1].index.intersection(perm_signatures[c2].index)
                    if len(common) < min_common_genes:
                        corr = np.nan
                    else:
                        corr, _ = spearmanr(perm_signatures[c1][common], perm_signatures[c2][common])
                    S_perm.loc[c1, c2] = corr
                    S_perm.loc[c2, c1] = corr

            D_perm = 1 - S_perm.fillna(1)
            np.fill_diagonal(D_perm.values, 0)
            Z_perm = linkage(squareform(D_perm.values), method=method)

            perm_labels = get_labels(Z_perm, k)
            # Stability: fraction of labels that match original clustering
            stabilities.append((perm_labels == base_labels[k]).mean())

        stability_results.loc[k, "mean_stability"] = np.mean(stabilities)

    return stability_results


# In[188]:


# Assume rank_signatures is a dict: {sample_name: pd.Series(rank)}
stability = permutation_cluster_significance(
    rank_signatures=rank_sigs,  # from your "up100" folder
    method="average",
    min_common_genes=int(0.05*100),  # 5% of 100 top genes = 5
    n_iter=100,
    max_k=10
)

print(stability)


# In[190]:


import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, set_link_color_palette

def plot_dendrogram_with_stability(Z, rank_signatures, stability_results, k=None, method="average", min_common_genes=5):
    """
    Plot a dendrogram with branch coloring based on permutation cluster stability.

    Z : linkage matrix
    rank_signatures : dict of sample_name -> pd.Series(rank)
    stability_results : pd.DataFrame with mean_stability per k
    k : int or None
        Number of clusters to highlight; if None, just plot dendrogram
    """
    cluster_names = list(rank_signatures.keys())
    
    plt.figure(figsize=(12,6))
    dendro = dendrogram(Z, labels=cluster_names, leaf_rotation=90, color_threshold=None)
    plt.title("Hierarchical clustering dendrogram")
    plt.ylabel("1 - Spearman correlation")
    
    if k is not None:
        # Compute cluster assignments
        labels = fcluster(Z, k, criterion="maxclust")
        label_map = {name: lbl for name, lbl in zip(cluster_names, labels)}
        
        # Color leaves by cluster
        leaf_colors = [label_map[leaf] for leaf in dendro['ivl']]
        cmap = plt.get_cmap("tab20")
        for leaf, x in zip(dendro['leaves'], leaf_colors):
            plt.plot([x, x], [0, 0.02], color=cmap((leaf-1) % 20), linewidth=5)  # small color tick
        
        # Overlay stability for this k
        mean_stab = stability_results.loc[k, "mean_stability"]
        plt.text(0.95, 0.95, f"Cluster stability (k={k}): {mean_stab:.2f}",
                 transform=plt.gca().transAxes, fontsize=12,
                 horizontalalignment="right", verticalalignment="top",
                 bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"))
    
    plt.tight_layout()
    plt.show()

# Example usage:
# Z, rank_signatures, stability_results from previous permutation test
plot_dendrogram_with_stability(Z, rank_sigs, stability_results, k=2)
plot_dendrogram_with_stability(Z, rank_sigs, stability_results, k=3)


# In[189]:


import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, set_link_color_palette

def plot_dendrogram_with_stability(Z, rank_signatures, stability_results, k=None, method="average", min_common_genes=5):
    """
    Plot a dendrogram with branch coloring based on permutation cluster stability.

    Z : linkage matrix
    rank_signatures : dict of sample_name -> pd.Series(rank)
    stability_results : pd.DataFrame with mean_stability per k
    k : int or None
        Number of clusters to highlight; if None, just plot dendrogram
    """
    cluster_names = list(rank_signatures.keys())
    
    plt.figure(figsize=(12,6))
    dendro = dendrogram(Z, labels=cluster_names, leaf_rotation=90, color_threshold=None)
    plt.title("Hierarchical clustering dendrogram")
    plt.ylabel("1 - Spearman correlation")
    
    if k is not None:
        # Compute cluster assignments
        labels = fcluster(Z, k, criterion="maxclust")
        label_map = {name: lbl for name, lbl in zip(cluster_names, labels)}
        
        # Color leaves by cluster
        leaf_colors = [label_map[leaf] for leaf in dendro['ivl']]
        cmap = plt.get_cmap("tab20")
        for leaf, x in zip(dendro['leaves'], leaf_colors):
            plt.plot([x, x], [0, 0.02], color=cmap((leaf-1) % 20), linewidth=5)  # small color tick
        
        # Overlay stability for this k
        mean_stab = stability_results.loc[k, "mean_stability"]
        plt.text(0.95, 0.95, f"Cluster stability (k={k}): {mean_stab:.2f}",
                 transform=plt.gca().transAxes, fontsize=12,
                 horizontalalignment="right", verticalalignment="top",
                 bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"))
    
    plt.tight_layout()
    plt.show()

# Example usage:
# Z, rank_signatures, stability_results from previous permutation test
plot_dendrogram_with_stability(Z, rank_sigs, stability_results, k=2)
plot_dendrogram_with_stability(Z, rank_sigs, stability_results, k=3)


# In[ ]:





# In[ ]:





# ## All Cell Types by Paper 

# In[ ]:


#test to see if epithelial cell populations from different papers cluster 
#by paper, or by cell type


# In[166]:


csv_directory_all_up20 = './data/All_epithelial/Up100/'
csv_files_all_up20 = glob.glob(csv_directory_all_up20 + '*.csv')
dfs_all_up20 = []

for file in csv_files_all_up20:
    df_all_up20 = pd.read_csv(file)
    df_all_up20 = df_all_up20[['names', 'logfoldchanges']]
    df_all_up20.set_index('names', inplace=True)
    file_name = file.split('/')[-1].replace('.csv', '')
    df_all_up20.columns = [file_name]
    dfs_all_up20.append(df_all_up20)

merged_df_all_up20 = pd.concat(dfs_all_up20, axis=1)
merged_df_all_up20.fillna(0, inplace=True)
gene_counts_all_up20 = (merged_df_all_up20 != 0).sum(axis=1)
# Keep only genes that appear in more than one file, use >1 !! HERE USING 0 !!
filtered_df_all_up20 = merged_df_all_up20[gene_counts_all_up20 > 0]

plt.figure(figsize=(60, 40))
sns.heatmap(filtered_df_all_up20, cmap='viridis', annot=True, fmt='.2f')
plt.title('Heatmap of Differential Gene Expression All Up20 (All, Not Genes in More Than One File)')
plt.xlabel('Files')
plt.ylabel('Genes')
plt.xticks(rotation=45)
plt.tight_layout()


# In[167]:


dist_matrix_all_cor_up20 = pdist(filtered_df_all_up20.T, metric='correlation')
dist_df_all_cor_up20 = pd.DataFrame(squareform(dist_matrix_all_cor_up20), index=filtered_df_all_up20.columns, columns=filtered_df_all_up20.columns)

linkage_matrix_all_cor_up20 = sns.clustermap(dist_df_all_cor_up20, annot=True, method='average', metric='correlation', figsize=(24, 16))


# In[168]:


df_all_up20 = merged_df_all_up20.copy()
df_transposed_all_up20 = df_all_up20.T
distance_matrix_all_up20 = pdist(df_transposed_all_up20, metric='correlation')
distance_matrix_square_all_up20 = squareform(distance_matrix_all_up20)

linkage_matrix_all_up20 = linkage(distance_matrix_all_up20, method='average')
max_d = 5  # Example threshold
clusters_all_up20 = fcluster(linkage_matrix_all_up20, max_d, criterion='distance')
df_transposed_all_up20['Cluster'] = clusters_all_up20

sns.clustermap(df_transposed_all_up20.drop(columns='Cluster'), 
               method='average', 
               metric='correlation', 
               figsize=(40, 40),
               cmap='viridis',
               row_cluster=False,  
               col_cluster=False)  


# In[169]:


plt.figure(figsize=(12, 12))
dendrogram(linkage_matrix_all_up20, labels=df_transposed_all_up20.index.tolist())
plt.title('Hierarchical Clustering Dendrogram All Up20')
plt.xlabel('Sample')
plt.ylabel('Distance')
plt.xticks(rotation=45, ha='right')
plt.show()


# ### Basal Example Run-Through Similarity Heatmaps ####

# In[285]:


csv_directory_basal_up30 = './data/basal/up30/'
csv_files_basal_up30 = glob.glob(csv_directory_basal_up30 + '*.csv')
dfs_basal_up30 = []


# In[286]:


for file in csv_files_basal_up30:
    df_basal_up30 = pd.read_csv(file)
    print(f"Columns in {file}: {df_basal_up30.columns}")


# In[287]:


for file in csv_files_basal_up30:
    df_basal_up30 = pd.read_csv(file)
    df_basal_up30 = df_basal_up30[['names', 'logfoldchanges']]
    df_basal_up30.set_index('names', inplace=True)
    file_name = file.split('/')[-1].replace('.csv', '')
    df_basal_up30.columns = [file_name]
    dfs_basal_up30.append(df_basal_up30)

merged_df_basal_up30 = pd.concat(dfs_basal_up30, axis=1)
merged_df_basal_up30.fillna(0, inplace=True)
gene_counts_basal_up30 = (merged_df_basal_up30 != 0).sum(axis=1)
# Keep only genes that appear in more than one file, use >1
filtered_df_basal_up30 = merged_df_basal_up30[gene_counts_basal_up30 > 1]


# In[288]:


plt.figure(figsize=(60, 40))
sns.heatmap(filtered_df_basal_up30, cmap='viridis', annot=True, fmt='.2f')
plt.title('Heatmap of Differential Gene Expression Basal Up30 (Genes in More Than One File)')
plt.xlabel('Files')
plt.ylabel('Genes')
plt.xticks(rotation=45)
plt.tight_layout()


# In[289]:


clustermap_basal_up30 = sns.clustermap(filtered_df_basal_up30, cmap='viridis', annot=True, fmt='.2f', figsize=(60, 40))


# In[290]:


dist_matrix_basal_up30 = pdist(filtered_df_basal_up30.T, metric='euclidean')
dist_df_basal_up30 = pd.DataFrame(squareform(dist_matrix_basal_up30), index=filtered_df_basal_up30.columns, columns=filtered_df_basal_up30.columns)


# In[291]:


linkage_matrix_basal_up30 = sns.clustermap(dist_df_basal_up30, annot=True, method='average', metric='euclidean', figsize=(24, 16))


# In[292]:


dist_matrix_basal_merged_up30 = pdist(merged_df_basal_up30.T, metric='euclidean')
dist_df_basal_merged_up30 = pd.DataFrame(squareform(dist_matrix_basal_merged_up30), index=merged_df_basal_up30.columns, columns=merged_df_basal_up30.columns)


# In[293]:


linkage_matrix_basal_merged_up30 = sns.clustermap(dist_df_basal_merged_up30, annot=True, method='average', metric='euclidean', figsize=(24, 16))


# In[294]:


dist_matrix_basal_cor_up30 = pdist(filtered_df_basal_up30.T, metric='correlation')
dist_df_basal_cor_up30 = pd.DataFrame(squareform(dist_matrix_basal_cor_up30), index=filtered_df_basal_up30.columns, columns=filtered_df_basal_up30.columns)


# In[295]:


linkage_matrix_basal_cor_up30 = sns.clustermap(dist_df_basal_cor_up30, annot=True, method='average', metric='correlation', figsize=(24, 16))


# In[296]:


# 🔹 Compute sample-to-sample correlation matrix
sample_correlation_basal_up30 = merged_df_basal_up30.corr(method='pearson')  # Compute correlation between samples

# 🔹 Convert to distance matrix for hierarchical clustering
distance_matrix_basal_up30 = pdist(sample_correlation_basal_up30, metric='euclidean')  # Use Euclidean for clustering
linkage_matrix_basal_up30 = sch.linkage(distance_matrix_basal_up30, method='average')  # Perform hierarchical clustering

# 🔹 Create a clustered heatmap of sample similarity
sns.clustermap(
    sample_correlation_basal_up30,  # Use the sample correlation matrix
    row_linkage=linkage_matrix_basal_up30,  # Cluster rows (samples)
    col_linkage=linkage_matrix_basal_up30,  # Cluster columns (samples)
    cmap='coolwarm',  # Red-blue colormap for similarity
    figsize=(30, 20),
    annot=True  # Set to False if you don't want numbers in the cells
)

plt.title("Sample Similarity Clustering Heatmap Basal Up30")
plt.show()


# In[297]:


dist_matrix_basal_cor_merged_up30 = pdist(merged_df_basal_up30.T, metric='correlation')
dist_df_basal_cor_merged_up30 = pd.DataFrame(squareform(dist_matrix_basal_cor_merged_up30), index=merged_df_basal_up30.columns, columns=merged_df_basal_up30.columns)


# In[298]:


linkage_matrix_basal_cor_merged_up30 = sns.clustermap(dist_df_basal_cor_merged_up30, annot=True, method='average', metric='correlation', figsize=(24, 16))


# In[299]:


#Computes a pairwise correlation distance matrix for dendrogram heirarchial clustering
df_b2_up30 = merged_df_basal_up30.copy()
df_transposed_b2_up30 = df_b2_up30.T
distance_matrix_b2_up30 = pdist(df_transposed_b2_up30, metric='correlation')
distance_matrix_square_b2_up30 = squareform(distance_matrix_b2_up30)


# In[300]:


linkage_matrix_b2_up30 = linkage(distance_matrix_b2_up30, method='average')
max_d = 5  # Example threshold
clusters_b2_up30 = fcluster(linkage_matrix_b2_up30, max_d, criterion='distance')
df_transposed_b2_up30['Cluster'] = clusters_b2_up30


# In[301]:


sns.clustermap(df_transposed_b2_up30.drop(columns='Cluster'), 
               method='average', 
               metric='correlation', 
               figsize=(40, 40),
               cmap='viridis',
               row_cluster=False,  
               col_cluster=False)  


# In[302]:


plt.figure(figsize=(12, 12))
dendrogram(linkage_matrix_b2_up30, labels=df_transposed_b2_up30.index.tolist())
plt.title('Hierarchical Clustering Dendrogram B2 Up30')
plt.xlabel('Sample')
plt.ylabel('Distance')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[303]:


#fixing correlation values based on distance (2-0) to compute pearson correlation directly (-1 to +1)#
correlation_matrix_fixed_basal_up30 = merged_df_basal_up30.corr(method='pearson')
print(correlation_matrix_fixed_basal_up30)


# In[304]:


plt.figure(figsize=(25, 15)) 
sns.heatmap(correlation_matrix_fixed_basal_up30, 
            annot=True,          
            cmap='viridis',      
            vmin=-1, vmax=1,     
            center=0,            
            linewidths=0.5,       
            linecolor='gray')


# In[305]:


clustermap_basal_corr_up30 = sns.clustermap(correlation_matrix_fixed_basal_up30, cmap='viridis', annot=True, fmt='.2f', figsize=(60, 40))


# ## Similar Basal Clusters ####

# In[ ]:


# === SETTINGS ===
input_folder = './data/basal/up100/'
input_folder_up1000 = './data/basal/up1000/'
output_folder = './data/basal/Heatmaps_up100/'
os.makedirs(output_folder, exist_ok=True)

groups = {
    "Group1": ["up100_JB2022_basal_1.csv", "up100_KK18_basal_2.csv"],
    "Group2": ["up100_KK18_basal_0.csv", "up100_JV2021_basal_2.csv"],
    "Group3": ["up100_JV2021_basal_1.csv", "up100_KN23_basal_1.csv"],
    "Group4": ["up100_KN23_basal_2.csv", "up100_JB2022_basal_1.csv", "up100_KK18_basal_2.csv", "up100_KK18_basal_0.csv", "up100_JV2021_basal_2.csv", "up100_JV2021_basal_1.csv", "up100_KN23_basal_1.csv"],
    "Group5": ["up100_KN23_basal_5.csv", "up100_JB2022_basal_3.csv"],
    "Group6": ["up100_KK18_basal_4.csv", "up100_KN23_basal_4.csv"],
    "Group7": ["up100_KK18_basal_5.csv", "up100_JB2022_basal_2.csv"],
    "Group8": ["up100_KK18_basal_4.csv", "up100_KN23_basal_4.csv", "up100_KK18_basal_5.csv", "up100_JB2022_basal_2.csv"],
    "Group9": ["up100_R2024_basal_2.csv", "up100_JV2021_basal_0.csv"],
    "Group10": ["up100_NN2023_basal_1.csv", "up100_N2021_basal_2.csv", "up100_KK18_basal_4.csv", "up100_KN23_basal_4.csv", "up100_KK18_basal_5.csv", "up100_JB2022_basal_2.csv", "up100_R2024_basal_2.csv", "up100_JV2021_basal_0.csv"],
    "Group11": ["up100_JB2022_basal_0.csv", "up100_N2021_basal_0.csv"],
    "Group12": ["up100_R2024_basal_3.csv", "up100_JB2022_basal_0.csv", "up100_N2021_basal_0.csv"],
    "Group13": ["up100_R2024_basal_0.csv", "up100_KN23_basal_3.csv"],
    "Group14": ["up100_NN2023_basal_0.csv", "up100_R2024_basal_0.csv", "up100_KN23_basal_3.csv"],
    "Group15": ["up100_N2021_basal_1.csv", "up100_JB2022_basal_4.csv"],
    "Group16": ["up100_JB2022_basal_5.csv", "up100_R2024_basal_1.csv"],
    "Group17": ["up100_N2021_basal_1.csv", "up100_JB2022_basal_4.csv", "up100_JB2022_basal_5.csv", "up100_R2024_basal_1.csv"]
}

# === COLOR MAP ===
fuchsia_cmap = LinearSegmentedColormap.from_list("fuchsia", ["#ffe6f9", "#cc3399", "#800055"])

# === HELPERS ===
def read_and_process_files(file_list, folder):
    dfs = []
    for file in file_list:
        path = os.path.join(folder, file)
        df = pd.read_csv(path, usecols=["names", "logfoldchanges"])
        df = df.rename(columns={"logfoldchanges": "logFC"})
        df["sample"] = file
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def fill_missing_from_up1000(heatmap_data, samples, folder_up1000):
    filled = heatmap_data.copy()
    for sample in samples:
        up1000_file = sample.replace("up100_", "up1000_")
        up1000_path = os.path.join(folder_up1000, up1000_file)
        up1000_df = pd.read_csv(up1000_path, usecols=["names", "logfoldchanges"]).set_index("names")
        for gene in filled.index:
            if pd.isna(filled.loc[gene, sample]):
                if gene in up1000_df.index:
                    filled.loc[gene, sample] = up1000_df.loc[gene, "logfoldchanges"]
    return filled

def zscore_per_column(df):
    return df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

from matplotlib.colors import LinearSegmentedColormap

# Diverging colormap: teal (neg) → white (zero) → fuchsia (pos)
div_cmap = LinearSegmentedColormap.from_list(
    "teal_fuchsia", ["#008080", "#ffffff", "#800055"]
)

# === MAIN LOOP ===
for group, samples in groups.items():
    # Read top100 files
    group_data = read_and_process_files(samples, input_folder)

    # Pivot to matrix
    heatmap_data = group_data.pivot_table(index="names", columns="sample", values="logFC", aggfunc="mean")

    # Fill missing genes from up1000
    heatmap_data_filled = fill_missing_from_up1000(heatmap_data, samples, input_folder_up1000)

    # Full heatmap (filled & z-scored) with diverging colors
    plt.figure(figsize=(10, 28))
    # Filter for overlapping genes (appear in >1 sample)
    gene_counts = group_data["names"].value_counts()
    filtered_genes = gene_counts[gene_counts > 1].index
    heatmap_data_filled_filtered = heatmap_data_filled.loc[heatmap_data_filled.index.isin(filtered_genes)]

    if heatmap_data_filled_filtered.empty:
        print(f"No overlapping genes for group {group}, skipping heatmap.")
    else:
        sns.heatmap(
            heatmap_data_filled_filtered,
            cmap=div_cmap,
            center=0,  # ensures zero is in the middle (white)
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC_filtered.png")
        plt.close()

    # Heatmap of all filled genes (without filtering)
    if heatmap_data_filled.empty:
        print(f"No genes to plot for group {group}, skipping full heatmap.")
    else:
        plt.figure(figsize=(10, 28))
        sns.heatmap(
            heatmap_data_filled,
            cmap=div_cmap,
            center=0,
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC.png")
        plt.close()


# In[5]:


def run_go_enrichment(sample_groups, data_dir, output_dir, gene_sets='GO_Biological_Process_2021', organism='Human'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for group_name, selected_samples in sample_groups.items():
        gene_data = []
        
        # Process each sample in the selected group
        for sample in selected_samples:
            file_path = os.path.join(data_dir, sample)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"Processing file: {file_path}")
                
                if 'names' in df.columns and 'logfoldchanges' in df.columns:
                    # Filter genes with positive logfold change
                    filtered_genes = df[df['logfoldchanges'] > 0]['names']
                    if not filtered_genes.empty:
                        gene_data.extend(filtered_genes)
                    else:
                        print(f"No genes passed the filter in {sample}")
                else:
                    print(f"Required columns are missing in {sample}")
            else:
                print(f"File {sample} not found!")
        
        # Remove duplicates
        unique_genes = list(set(gene_data))
        print(f"Unique Genes in {group_name}: {unique_genes}")
        
        if unique_genes:
            # Perform GO enrichment analysis
            results = enrichr(
                gene_list=unique_genes,
                gene_sets=gene_sets,  
                organism=organism
            )

            # Save the enrichment results to a CSV file
            output_file = os.path.join(output_dir, f"{group_name}_go_enrichment_results.csv")
            results.res2d.to_csv(output_file, index=False)
            print(f"Saved enrichment results for {group_name} to {output_file}")

            def style_ax(ax, title_size=60, label_size=45, tick_size=45):
                ax.set_title(ax.get_title(), fontsize=title_size, fontweight='bold')
                ax.set_xlabel(ax.get_xlabel(), fontsize=label_size, fontweight='bold')
                ax.set_ylabel(ax.get_ylabel(), fontsize=label_size, fontweight='bold')
                ax.tick_params(axis='both', which='major', labelsize=tick_size)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight('bold')

           # Barplot
            barplot_output = os.path.join(output_dir, f"{group_name}_go_enrichment_barplot.png")
            ax = barplot(results.res2d, title=f'{group_name} GO Enrichment Analysis', top_term=10, cmap='viridis', figsize=(70, 50))
            style_ax(ax)
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(barplot_output)
            plt.close(fig)

            # Dotplot
            dotplot_output = os.path.join(output_dir, f"{group_name}_go_enrichment_dotplot.png")
            ax = dotplot(results.res2d, title=f'{group_name} GO Enrichment Analysis (Dotplot)', top_term=10, figsize=(70, 50))
            style_ax(ax)
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(dotplot_output)
            plt.close(fig)
            print(f"Saved dotplot for {group_name} to {dotplot_output}")
        else:
            print(f"Gene list for {group_name} is empty. Ensure the input files have valid data.")

# Example of your sample groups
sample_groups = {
    "Group1": ["up1000_JB2022_basal_1.csv", "up1000_KK18_basal_2.csv"],
    "Group2": ["up1000_KK18_basal_0.csv", "up1000_JV2021_basal_2.csv"],
    "Group3": ["up1000_JV2021_basal_1.csv", "up1000_KN23_basal_1.csv"],
    "Group4": ["up1000_KN23_basal_2.csv", "up1000_JB2022_basal_1.csv", "up1000_KK18_basal_2.csv", "up1000_KK18_basal_0.csv", "up1000_JV2021_basal_2.csv", "up1000_JV2021_basal_1.csv", "up1000_KN23_basal_1.csv"],
    "Group5": ["up1000_KN23_basal_5.csv", "up1000_JB2022_basal_3.csv"],
    "Group6": ["up1000_KK18_basal_4.csv", "up1000_KN23_basal_4.csv"],
    "Group7": ["up1000_KK18_basal_5.csv", "up1000_JB2022_basal_2.csv"],
    "Group8": ["up1000_KK18_basal_4.csv", "up1000_KN23_basal_4.csv", "up1000_KK18_basal_5.csv", "up1000_JB2022_basal_2.csv"],
    "Group9": ["up1000_R2024_basal_2.csv", "up1000_JV2021_basal_0.csv"],
    "Group10": ["up1000_NN2023_basal_1.csv", "up1000_N2021_basal_2.csv", "up1000_KK18_basal_4.csv", "up1000_KN23_basal_4.csv", "up1000_KK18_basal_5.csv", "up1000_JB2022_basal_2.csv", "up1000_R2024_basal_2.csv", "up1000_JV2021_basal_0.csv"],
    "Group11": ["up1000_JB2022_basal_0.csv", "up1000_N2021_basal_0.csv"],
    "Group12": ["up1000_R2024_basal_3.csv", "up1000_JB2022_basal_0.csv", "up1000_N2021_basal_0.csv"],
    "Group13": ["up1000_R2024_basal_0.csv", "up1000_KN23_basal_3.csv"],
    "Group14": ["up1000_NN2023_basal_0.csv", "up1000_R2024_basal_0.csv", "up1000_KN23_basal_3.csv"],
    "Group15": ["up1000_N2021_basal_1.csv", "up1000_JB2022_basal_4.csv"],
    "Group16": ["up1000_JB2022_basal_5.csv", "up1000_R2024_basal_1.csv"],
    "Group17": ["up1000_N2021_basal_1.csv", "up1000_JB2022_basal_4.csv", "up1000_JB2022_basal_5.csv", "up1000_R2024_basal_1.csv"]
}

# Example directories (make sure to adjust these to your file locations)
data_dir = './data/basal/up100/'
output_dir = './data/basal/GO_Enrichment_Results/'

# Run GO enrichment for all groups and save results and plots
run_go_enrichment(sample_groups, data_dir, output_dir)


# In[4]:


sample_groups = {
    "Group1": ["up100_JB2022_basal_1.csv", "up100_KK18_basal_2.csv"],
    "Group2": ["up100_KK18_basal_0.csv", "up100_JV2021_basal_2.csv"],
    "Group3": ["up100_JV2021_basal_1.csv", "up100_KN23_basal_1.csv"],
    "Group4": ["up100_KN23_basal_2.csv", "up100_JB2022_basal_1.csv", "up100_KK18_basal_2.csv", "up100_KK18_basal_0.csv", "up100_JV2021_basal_2.csv", "up100_JV2021_basal_1.csv", "up100_KN23_basal_1.csv"],
    "Group5": ["up100_KN23_basal_5.csv", "up100_JB2022_basal_3.csv"],
    "Group6": ["up100_KK18_basal_4.csv", "up100_KN23_basal_4.csv"],
    "Group7": ["up100_KK18_basal_5.csv", "up100_JB2022_basal_2.csv"],
    "Group8": ["up100_KK18_basal_4.csv", "up100_KN23_basal_4.csv", "up100_KK18_basal_5.csv", "up100_JB2022_basal_2.csv"],
    "Group9": ["up100_R2024_basal_2.csv", "up100_JV2021_basal_0.csv"],
    "Group10": ["up100_NN2023_basal_1.csv", "up100_N2021_basal_2.csv", "up100_KK18_basal_4.csv", "up100_KN23_basal_4.csv", "up100_KK18_basal_5.csv", "up100_JB2022_basal_2.csv", "up100_R2024_basal_2.csv", "up100_JV2021_basal_0.csv"],
    "Group11": ["up100_JB2022_basal_0.csv", "up100_N2021_basal_0.csv"],
    "Group12": ["up100_R2024_basal_3.csv", "up100_JB2022_basal_0.csv", "up100_N2021_basal_0.csv"],
    "Group13": ["up100_R2024_basal_0.csv", "up100_KN23_basal_3.csv"],
    "Group14": ["up100_NN2023_basal_0.csv", "up100_R2024_basal_0.csv", "up100_KN23_basal_3.csv"],
    "Group15": ["up100_N2021_basal_1.csv", "up100_JB2022_basal_4.csv"],
    "Group16": ["up100_JB2022_basal_5.csv", "up100_R2024_basal_1.csv"],
    "Group17": ["up100_N2021_basal_1.csv", "up100_JB2022_basal_4.csv", "up100_JB2022_basal_5.csv", "up100_R2024_basal_1.csv"]
}

# Example directories (make sure to adjust these to your file locations)
data_dir = './data/basal/up100/'
output_dir = './data/basal/Intersect_GO_Enrichment_Results/'

# Run GO enrichment for all groups and save results and plots
run_go_enrichment_intersect(sample_groups, data_dir, output_dir)


# In[3]:


sample_groups = {
    "Group1": ["up1000_JB2022_basal_1.csv", "up1000_KK18_basal_2.csv"],
    "Group2": ["up1000_KK18_basal_0.csv", "up1000_JV2021_basal_2.csv"],
    "Group3": ["up1000_JV2021_basal_1.csv", "up1000_KN23_basal_1.csv"],
    "Group4": ["up1000_KN23_basal_2.csv", "up1000_JB2022_basal_1.csv", "up1000_KK18_basal_2.csv", "up1000_KK18_basal_0.csv", "up1000_JV2021_basal_2.csv", "up1000_JV2021_basal_1.csv", "up1000_KN23_basal_1.csv"],
    "Group5": ["up1000_KN23_basal_5.csv", "up1000_JB2022_basal_3.csv"],
    "Group6": ["up1000_KK18_basal_4.csv", "up1000_KN23_basal_4.csv"],
    "Group7": ["up1000_KK18_basal_5.csv", "up1000_JB2022_basal_2.csv"],
    "Group8": ["up1000_KK18_basal_4.csv", "up1000_KN23_basal_4.csv", "up1000_KK18_basal_5.csv", "up1000_JB2022_basal_2.csv"],
    "Group9": ["up1000_R2024_basal_2.csv", "up1000_JV2021_basal_0.csv"],
    "Group10": ["up1000_NN2023_basal_1.csv", "up1000_N2021_basal_2.csv", "up1000_KK18_basal_4.csv", "up1000_KN23_basal_4.csv", "up1000_KK18_basal_5.csv", "up1000_JB2022_basal_2.csv", "up1000_R2024_basal_2.csv", "up1000_JV2021_basal_0.csv"],
    "Group11": ["up1000_JB2022_basal_0.csv", "up1000_N2021_basal_0.csv"],
    "Group12": ["up1000_R2024_basal_3.csv", "up1000_JB2022_basal_0.csv", "up1000_N2021_basal_0.csv"],
    "Group13": ["up1000_R2024_basal_0.csv", "up1000_KN23_basal_3.csv"],
    "Group14": ["up1000_NN2023_basal_0.csv", "up1000_R2024_basal_0.csv", "up1000_KN23_basal_3.csv"],
    "Group15": ["up1000_N2021_basal_1.csv", "up1000_JB2022_basal_4.csv"],
    "Group16": ["up1000_JB2022_basal_5.csv", "up1000_R2024_basal_1.csv"],
    "Group17": ["up1000_N2021_basal_1.csv", "up1000_JB2022_basal_4.csv", "up1000_JB2022_basal_5.csv", "up1000_R2024_basal_1.csv"]
}

# Example directories (make sure to adjust these to your file locations)
data_dir = './data/basal/up1000/'
output_dir = './data/basal/1000_Intersect_GO_Enrichment_Results/'

# Run GO enrichment for all groups and save results and plots
run_go_enrichment_intersect(sample_groups, data_dir, output_dir)


# ## Similar LM Clusters ###

# In[203]:


# === SETTINGS ===
input_folder = './data/mature_luminal/up100/'
input_folder_up1000 = './data/mature_luminal/up1000/'
output_folder = './data/mature_luminal/Heatmaps_up100/'
os.makedirs(output_folder, exist_ok=True)

groups = {
    "Group1": ["up100_JV2021_mature_luminal_2.csv", "up100_JB2022_mature_luminal_1.csv"],
    "Group2": ["up100_N2021_mature_luminal_0.csv", "up100_JV2021_mature_luminal_2.csv", "up100_JB2022_mature_luminal_1.csv"],
    "Group3": ["up100_KK18_mature_luminal_0.csv", "up100_N2021_mature_luminal_0.csv", "up100_JV2021_mature_luminal_2.csv", "up100_JB2022_mature_luminal_1.csv"],
    "Group4": ["up100_N2021_mature_luminal_1.csv", "up100_JB2022_mature_luminal_0.csv"],
    "Group5": ["up100_KN23_mature_luminal_3.csv", "up100_JV2021_mature_luminal_0.csv"],
    "Group6": ["up100_N2021_mature_luminal_1.csv", "up100_JB2022_mature_luminal_0.csv", "up100_KN23_mature_luminal_3.csv", "up100_JV2021_mature_luminal_0.csv"],
    "Group7": ["up100_KN23_mature_luminal_2.csv", "up100_R2024_mature_luminal_0.csv"],
    "Group8": ["up100_JV2021_mature_luminal_1.csv", "up100_KN23_mature_luminal_1.csv"],
    "Group9": ["up100_NN2023_mature_luminal_1.csv", "up100_JV2021_mature_luminal_1.csv", "up100_KN23_mature_luminal_1.csv"],
    "Group10": ["up100_KN23_mature_luminal_2.csv", "up100_R2024_mature_luminal_0.csv", "up100_NN2023_mature_luminal_1.csv", "up100_JV2021_mature_luminal_1.csv", "up100_KN23_mature_luminal_1.csv"],
    "Group11": ["up100_KK18_mature_luminal_2.csv", "up100_KN23_mature_luminal_0.csv"],
    "Group12": ["up100_KK18_mature_luminal_2.csv", "up100_KN23_mature_luminal_0.csv", "up100_KN23_mature_luminal_4.csv"],
    "Group13": ["up100_NN2023_mature_luminal_0.csv", "up100_JB2022_mature_luminal_2.csv"],
    "Group14": ["up100_R2024_mature_luminal_1.csv", "up100_R2024_mature_luminal_2.csv", "up100_NN2023_mature_luminal_0.csv", "up100_JB2022_mature_luminal_2.csv"],
    "Group15": ["up100_JB2022_mature_luminal_4.csv", "up100_JB2022_mature_luminal_3.csv", "up100_N2021_mature_luminal_2.csv"]
}

# === COLOR MAP ===
fuchsia_cmap = LinearSegmentedColormap.from_list("fuchsia", ["#ffe6f9", "#cc3399", "#800055"])

# === HELPERS ===
def read_and_process_files(file_list, folder):
    dfs = []
    for file in file_list:
        path = os.path.join(folder, file)
        df = pd.read_csv(path, usecols=["names", "logfoldchanges"])
        df = df.rename(columns={"logfoldchanges": "logFC"})
        df["sample"] = file
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def fill_missing_from_up1000(heatmap_data, samples, folder_up1000):
    filled = heatmap_data.copy()
    for sample in samples:
        up1000_file = sample.replace("up100_", "up1000_")
        up1000_path = os.path.join(folder_up1000, up1000_file)
        up1000_df = pd.read_csv(up1000_path, usecols=["names", "logfoldchanges"]).set_index("names")
        for gene in filled.index:
            if pd.isna(filled.loc[gene, sample]):
                if gene in up1000_df.index:
                    filled.loc[gene, sample] = up1000_df.loc[gene, "logfoldchanges"]
    return filled

from matplotlib.colors import LinearSegmentedColormap

# Diverging colormap: teal (neg) → white (zero) → fuchsia (pos)
div_cmap = LinearSegmentedColormap.from_list(
    "teal_fuchsia", ["#008080", "#ffffff", "#800055"]
)

for group, samples in groups.items():
    # Read top100 files
    group_data = read_and_process_files(samples, input_folder)

    # Pivot to matrix
    heatmap_data = group_data.pivot_table(index="names", columns="sample", values="logFC", aggfunc="mean")

    # Fill missing genes from up1000
    heatmap_data_filled = fill_missing_from_up1000(heatmap_data, samples, input_folder_up1000)

    # Full heatmap (filled & z-scored) with diverging colors
    plt.figure(figsize=(10, 28))
    # Filter for overlapping genes (appear in >1 sample)
    gene_counts = group_data["names"].value_counts()
    filtered_genes = gene_counts[gene_counts > 1].index
    heatmap_data_filled_filtered = heatmap_data_filled.loc[heatmap_data_filled.index.isin(filtered_genes)]

    if heatmap_data_filled_filtered.empty:
        print(f"No overlapping genes for group {group}, skipping heatmap.")
    else:
        sns.heatmap(
            heatmap_data_filled_filtered,
            cmap=div_cmap,
            center=0,  # ensures zero is in the middle (white)
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC_filtered.png")
        plt.close()

    # Heatmap of all filled genes (without filtering)
    if heatmap_data_filled.empty:
        print(f"No genes to plot for group {group}, skipping full heatmap.")
    else:
        plt.figure(figsize=(10, 28))
        sns.heatmap(
            heatmap_data_filled,
            cmap=div_cmap,
            center=0,
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC.png")
        plt.close()


# In[212]:


def run_go_enrichment(sample_groups, data_dir, output_dir, gene_sets='GO_Biological_Process_2021', organism='Human'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for group_name, selected_samples in sample_groups.items():
        gene_data = []
        
        # Process each sample in the selected group
        for sample in selected_samples:
            file_path = os.path.join(data_dir, sample)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"Processing file: {file_path}")
                
                if 'names' in df.columns and 'logfoldchanges' in df.columns:
                    # Filter genes with positive logfold change
                    filtered_genes = df[df['logfoldchanges'] > 0]['names']
                    if not filtered_genes.empty:
                        gene_data.extend(filtered_genes)
                    else:
                        print(f"No genes passed the filter in {sample}")
                else:
                    print(f"Required columns are missing in {sample}")
            else:
                print(f"File {sample} not found!")
        
        # Remove duplicates
        unique_genes = list(set(gene_data))
        print(f"Unique Genes in {group_name}: {unique_genes}")
        
        if unique_genes:
            # Perform GO enrichment analysis
            results = enrichr(
                gene_list=unique_genes,
                gene_sets=gene_sets,  
                organism=organism
            )

            # Save the enrichment results to a CSV file
            output_file = os.path.join(output_dir, f"{group_name}_go_enrichment_results.csv")
            results.res2d.to_csv(output_file, index=False)
            print(f"Saved enrichment results for {group_name} to {output_file}")

            def style_ax(ax, title_size=60, label_size=45, tick_size=45):
                ax.set_title(ax.get_title(), fontsize=title_size, fontweight='bold')
                ax.set_xlabel(ax.get_xlabel(), fontsize=label_size, fontweight='bold')
                ax.set_ylabel(ax.get_ylabel(), fontsize=label_size, fontweight='bold')
                ax.tick_params(axis='both', which='major', labelsize=tick_size)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight('bold')

           # Barplot
            barplot_output = os.path.join(output_dir, f"{group_name}_go_enrichment_barplot.png")
            ax = barplot(results.res2d, title=f'{group_name} GO Enrichment Analysis', top_term=10, cmap='viridis', figsize=(70, 50))
            style_ax(ax)
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(barplot_output)
            plt.close(fig)

            # Dotplot
            dotplot_output = os.path.join(output_dir, f"{group_name}_go_enrichment_dotplot.png")
            ax = dotplot(results.res2d, title=f'{group_name} GO Enrichment Analysis (Dotplot)', top_term=10, figsize=(70, 50))
            style_ax(ax)
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(dotplot_output)
            plt.close(fig)
            print(f"Saved dotplot for {group_name} to {dotplot_output}")
        else:
            print(f"Gene list for {group_name} is empty. Ensure the input files have valid data.")

# Example of your sample groups
sample_groups = {
    "Group1": ["up100_JV2021_mature_luminal_2.csv", "up100_JB2022_mature_luminal_1.csv"],
    "Group2": ["up100_N2021_mature_luminal_0.csv", "up100_JV2021_mature_luminal_2.csv", "up100_JB2022_mature_luminal_1.csv"],
    "Group3": ["up100_KK18_mature_luminal_0.csv", "up100_N2021_mature_luminal_0.csv", "up100_JV2021_mature_luminal_2.csv", "up100_JB2022_mature_luminal_1.csv"],
    "Group4": ["up100_N2021_mature_luminal_1.csv", "up100_JB2022_mature_luminal_0.csv"],
    "Group5": ["up100_KN23_mature_luminal_3.csv", "up100_JV2021_mature_luminal_0.csv"],
    "Group6": ["up100_N2021_mature_luminal_1.csv", "up100_JB2022_mature_luminal_0.csv", "up100_KN23_mature_luminal_3.csv", "up100_JV2021_mature_luminal_0.csv"],
    "Group7": ["up100_KN23_mature_luminal_2.csv", "up100_R2024_mature_luminal_0.csv"],
    "Group8": ["up100_JV2021_mature_luminal_1.csv", "up100_KN23_mature_luminal_1.csv"],
    "Group9": ["up100_NN2023_mature_luminal_1.csv", "up100_JV2021_mature_luminal_1.csv", "up100_KN23_mature_luminal_1.csv"],
    "Group10": ["up100_KN23_mature_luminal_2.csv", "up100_R2024_mature_luminal_0.csv", "up100_NN2023_mature_luminal_1.csv", "up100_JV2021_mature_luminal_1.csv", "up100_KN23_mature_luminal_1.csv"],
    "Group11": ["up100_KK18_mature_luminal_2.csv", "up100_KN23_mature_luminal_0.csv"],
    "Group12": ["up100_KK18_mature_luminal_2.csv", "up100_KN23_mature_luminal_0.csv", "up100_KN23_mature_luminal_4.csv"],
    "Group13": ["up100_NN2023_mature_luminal_0.csv", "up100_JB2022_mature_luminal_2.csv"],
    "Group14": ["up100_R2024_mature_luminal_1.csv", "up100_R2024_mature_luminal_2.csv", "up100_NN2023_mature_luminal_0.csv", "up100_JB2022_mature_luminal_2.csv"],
    "Group15": ["up100_JB2022_mature_luminal_4.csv", "up100_JB2022_mature_luminal_3.csv", "up100_N2021_mature_luminal_2.csv"]
}

# Example directories (make sure to adjust these to your file locations)
data_dir = './data/mature_luminal/up100/'
output_dir = './data/mature_luminal/GO_Enrichment_Results/'

# Run GO enrichment for all groups and save results and plots
run_go_enrichment(sample_groups, data_dir, output_dir)


# In[214]:


sample_groups = {
    "Group1": ["up100_JV2021_mature_luminal_2.csv", "up100_JB2022_mature_luminal_1.csv"],
    "Group2": ["up100_N2021_mature_luminal_0.csv", "up100_JV2021_mature_luminal_2.csv", "up100_JB2022_mature_luminal_1.csv"],
    "Group3": ["up100_KK18_mature_luminal_0.csv", "up100_N2021_mature_luminal_0.csv", "up100_JV2021_mature_luminal_2.csv", "up100_JB2022_mature_luminal_1.csv"],
    "Group4": ["up100_N2021_mature_luminal_1.csv", "up100_JB2022_mature_luminal_0.csv"],
    "Group5": ["up100_KN23_mature_luminal_3.csv", "up100_JV2021_mature_luminal_0.csv"],
    "Group6": ["up100_N2021_mature_luminal_1.csv", "up100_JB2022_mature_luminal_0.csv", "up100_KN23_mature_luminal_3.csv", "up100_JV2021_mature_luminal_0.csv"],
    "Group7": ["up100_KN23_mature_luminal_2.csv", "up100_R2024_mature_luminal_0.csv"],
    "Group8": ["up100_JV2021_mature_luminal_1.csv", "up100_KN23_mature_luminal_1.csv"],
    "Group9": ["up100_NN2023_mature_luminal_1.csv", "up100_JV2021_mature_luminal_1.csv", "up100_KN23_mature_luminal_1.csv"],
    "Group10": ["up100_KN23_mature_luminal_2.csv", "up100_R2024_mature_luminal_0.csv", "up100_NN2023_mature_luminal_1.csv", "up100_JV2021_mature_luminal_1.csv", "up100_KN23_mature_luminal_1.csv"],
    "Group11": ["up100_KK18_mature_luminal_2.csv", "up100_KN23_mature_luminal_0.csv"],
    "Group12": ["up100_KK18_mature_luminal_2.csv", "up100_KN23_mature_luminal_0.csv", "up100_KN23_mature_luminal_4.csv"],
    "Group13": ["up100_NN2023_mature_luminal_0.csv", "up100_JB2022_mature_luminal_2.csv"],
    "Group14": ["up100_R2024_mature_luminal_1.csv", "up100_R2024_mature_luminal_2.csv", "up100_NN2023_mature_luminal_0.csv", "up100_JB2022_mature_luminal_2.csv"],
    "Group15": ["up100_JB2022_mature_luminal_4.csv", "up100_JB2022_mature_luminal_3.csv", "up100_N2021_mature_luminal_2.csv"]
}

# Example directories (make sure to adjust these to your file locations)
data_dir = './data/mature_luminal/up100/'
output_dir = './data/mature_luminal/Intersect_GO_Enrichment_Results/'

# Run GO enrichment for all groups and save results and plots
run_go_enrichment_intersect(sample_groups, data_dir, output_dir)


# In[215]:


sample_groups = {
    "Group1": ["up1000_JV2021_mature_luminal_2.csv", "up1000_JB2022_mature_luminal_1.csv"],
    "Group2": ["up1000_N2021_mature_luminal_0.csv", "up1000_JV2021_mature_luminal_2.csv", "up1000_JB2022_mature_luminal_1.csv"],
    "Group3": ["up1000_KK18_mature_luminal_0.csv", "up1000_N2021_mature_luminal_0.csv", "up1000_JV2021_mature_luminal_2.csv", "up1000_JB2022_mature_luminal_1.csv"],
    "Group4": ["up1000_N2021_mature_luminal_1.csv", "up1000_JB2022_mature_luminal_0.csv"],
    "Group5": ["up1000_KN23_mature_luminal_3.csv", "up1000_JV2021_mature_luminal_0.csv"],
    "Group6": ["up1000_N2021_mature_luminal_1.csv", "up1000_JB2022_mature_luminal_0.csv", "up1000_KN23_mature_luminal_3.csv", "up1000_JV2021_mature_luminal_0.csv"],
    "Group7": ["up1000_KN23_mature_luminal_2.csv", "up1000_R2024_mature_luminal_0.csv"],
    "Group8": ["up1000_JV2021_mature_luminal_1.csv", "up1000_KN23_mature_luminal_1.csv"],
    "Group9": ["up1000_NN2023_mature_luminal_1.csv", "up1000_JV2021_mature_luminal_1.csv", "up1000_KN23_mature_luminal_1.csv"],
    "Group10": ["up1000_KN23_mature_luminal_2.csv", "up1000_R2024_mature_luminal_0.csv", "up1000_NN2023_mature_luminal_1.csv", "up1000_JV2021_mature_luminal_1.csv", "up1000_KN23_mature_luminal_1.csv"],
    "Group11": ["up1000_KK18_mature_luminal_2.csv", "up1000_KN23_mature_luminal_0.csv"],
    "Group12": ["up1000_KK18_mature_luminal_2.csv", "up1000_KN23_mature_luminal_0.csv", "up1000_KN23_mature_luminal_4.csv"],
    "Group13": ["up1000_NN2023_mature_luminal_0.csv", "up1000_JB2022_mature_luminal_2.csv"],
    "Group14": ["up1000_R2024_mature_luminal_1.csv", "up1000_R2024_mature_luminal_2.csv", "up1000_NN2023_mature_luminal_0.csv", "up1000_JB2022_mature_luminal_2.csv"],
    "Group15": ["up1000_JB2022_mature_luminal_4.csv", "up1000_JB2022_mature_luminal_3.csv", "up1000_N2021_mature_luminal_2.csv"]
}

# Example directories (make sure to adjust these to your file locations)
data_dir = './data/mature_luminal/up1000/'
output_dir = './data/mature_luminal/1000_Intersect_GO_Enrichment_Results/'

# Run GO enrichment for all groups and save results and plots
run_go_enrichment_intersect(sample_groups, data_dir, output_dir)


# ### LP Similar Clusters

# In[225]:


# === SETTINGS ===
input_folder = './data/luminal_progenitor/up100/'
input_folder_up1000 = './data/luminal_progenitor/up1000/'
output_folder = './data/luminal_progenitor/Heatmaps_up100/'
os.makedirs(output_folder, exist_ok=True)

groups = {
    "Group1": ["up100_NN2023_luminal_progenitor_0.csv", "up100_N2021_luminal_progenitor_6.csv"],
    "Group2": ["up100_N2021_luminal_progenitor_1.csv", "up100_R2024_luminal_progenitor_1.csv"],
    "Group3": ["up100_KN23_luminal_progenitor_4.csv", "up100_KK18_luminal_progenitor_1.csv"],
    "Group4": ["up100_N2021_luminal_progenitor_1.csv", "up100_R2024_luminal_progenitor_1.csv", "up100_KN23_luminal_progenitor_4.csv", "up100_KK18_luminal_progenitor_1.csv"],
    "Group5": ["up100_N2021_luminal_progenitor_1.csv", "up100_R2024_luminal_progenitor_1.csv", "up100_KN23_luminal_progenitor_4.csv", "up100_KK18_luminal_progenitor_1.csv", "up100_NN2023_luminal_progenitor_0.csv", "up100_N2021_luminal_progenitor_6.csv"],
    "Group6": ["up100_KN23_luminal_progenitor_1.csv", "up100_JV2021_luminal_progenitor_1.csv"],
    "Group7": ["up100_N2021_luminal_progenitor_3.csv", "up100_R2024_luminal_progenitor_0.csv"],
    "Group8": ["up100_KK18_luminal_progenitor_3.csv", "up100_N2021_luminal_progenitor_3.csv", "up100_R2024_luminal_progenitor_0.csv"],
    "Group9": ["up100_JV2021_luminal_progenitor_2.csv", "up100_NN2023_luminal_progenitor_1.csv"],
    "Group10": ["up100_KK18_luminal_progenitor_0.csv", "up100_KN23_luminal_progenitor_3.csv", "up100_KK18_luminal_progenitor_0.csv", "up100_JV2021_luminal_progenitor_2.csv", "up100_NN2023_luminal_progenitor_1.csv"],
    "Group11": ["up100_KN23_luminal_progenitor_3.csv", "up100_KK18_luminal_progenitor_0.csv", "up100_JV2021_luminal_progenitor_2.csv", "up100_NN2023_luminal_progenitor_1.csv", "up100_KK18_luminal_progenitor_3.csv", "up100_N2021_luminal_progenitor_3.csv", "up100_R2024_luminal_progenitor_0.csv", "up100_KN23_luminal_progenitor_1.csv", "up100_JV2021_luminal_progenitor_1.csv"],
    "Group12": ["up100_N2021_luminal_progenitor_5.csv", "up100_KN23_luminal_progenitor_0.csv"],
    "Group13": ["up100_JV2021_luminal_progenitor_3.csv", "up100_N2021_luminal_progenitor_2.csv"],
    "Group14": ["up100_N2021_luminal_progenitor_7.csv", "up100_KK18_luminal_progenitor_2.csv"],
    "Group15": ["up100_JV2021_luminal_progenitor_3.csv", "up100_N2021_luminal_progenitor_2.csv", "up100_N2021_luminal_progenitor_7.csv", "up100_KK18_luminal_progenitor_2.csv"],
    "Group16": ["up100_JV2021_luminal_progenitor_3.csv", "up100_N2021_luminal_progenitor_2.csv", "up100_N2021_luminal_progenitor_7.csv", "up100_KK18_luminal_progenitor_2.csv", "up100_N2021_luminal_progenitor_5.csv", "up100_KN23_luminal_progenitor_0.csv"],
    "Group17": ["up100_N2021_luminal_progenitor_0.csv", "up100_KN23_luminal_progenitor_2.csv"],
    "Group18": ["up100_KK18_luminal_progenitor_4.csv", "up100_JV2021_luminal_progenitor_4.csv"],
    "Group19": ["up100_R2024_luminal_progenitor_2.csv", "up100_KK18_luminal_progenitor_4.csv", "up100_JV2021_luminal_progenitor_4.csv"],
    "Group20": ["up100_JV2021_luminal_progenitor_3.csv", "up100_N2021_luminal_progenitor_2.csv", "up100_N2021_luminal_progenitor_7.csv", "up100_KK18_luminal_progenitor_2.csv", "up100_N2021_luminal_progenitor_5.csv", "up100_KN23_luminal_progenitor_0.csv", "up100_N2021_luminal_progenitor_0.csv", "up100_KN23_luminal_progenitor_2.csv", "up100_R2024_luminal_progenitor_2.csv", "up100_KK18_luminal_progenitor_4.csv", "up100_JV2021_luminal_progenitor_4.csv"],
    "Group21": ["up100_JV2021_luminal_progenitor_0.csv", "up100_JB2022_luminal_progenitor_2.csv"],
    "Group22": ["up100_N2021_luminal_progenitor_4.csv", "up100_JV2021_luminal_progenitor_0.csv", "up100_JB2022_luminal_progenitor_2.csv"]
}

# === COLOR MAP ===
fuchsia_cmap = LinearSegmentedColormap.from_list("fuchsia", ["#ffe6f9", "#cc3399", "#800055"])

# === HELPERS ===
def read_and_process_files(file_list, folder):
    dfs = []
    for file in file_list:
        path = os.path.join(folder, file)
        df = pd.read_csv(path, usecols=["names", "logfoldchanges"])
        df = df.rename(columns={"logfoldchanges": "logFC"})
        df["sample"] = file
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def fill_missing_from_up1000(heatmap_data, samples, folder_up1000):
    filled = heatmap_data.copy()
    for sample in samples:
        up1000_file = sample.replace("up100_", "up1000_")
        up1000_path = os.path.join(folder_up1000, up1000_file)
        up1000_df = pd.read_csv(up1000_path, usecols=["names", "logfoldchanges"]).set_index("names")
        for gene in filled.index:
            if pd.isna(filled.loc[gene, sample]):
                if gene in up1000_df.index:
                    filled.loc[gene, sample] = up1000_df.loc[gene, "logfoldchanges"]
    return filled

from matplotlib.colors import LinearSegmentedColormap

# Diverging colormap: teal (neg) → white (zero) → fuchsia (pos)
div_cmap = LinearSegmentedColormap.from_list(
    "teal_fuchsia", ["#008080", "#ffffff", "#800055"]
)

for group, samples in groups.items():
    # Read top100 files
    group_data = read_and_process_files(samples, input_folder)

    # Pivot to matrix
    heatmap_data = group_data.pivot_table(index="names", columns="sample", values="logFC", aggfunc="mean")

    # Fill missing genes from up1000
    heatmap_data_filled = fill_missing_from_up1000(heatmap_data, samples, input_folder_up1000)

    # Full heatmap (filled & z-scored) with diverging colors
    plt.figure(figsize=(10, 28))
    # Filter for overlapping genes (appear in >1 sample)
    gene_counts = group_data["names"].value_counts()
    filtered_genes = gene_counts[gene_counts > 1].index
    heatmap_data_filled_filtered = heatmap_data_filled.loc[heatmap_data_filled.index.isin(filtered_genes)]

    if heatmap_data_filled_filtered.empty:
        print(f"No overlapping genes for group {group}, skipping heatmap.")
    else:
        sns.heatmap(
            heatmap_data_filled_filtered,
            cmap=div_cmap,
            center=0,  # ensures zero is in the middle (white)
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC_filtered.png")
        plt.close()

    # Heatmap of all filled genes (without filtering)
    if heatmap_data_filled.empty:
        print(f"No genes to plot for group {group}, skipping full heatmap.")
    else:
        plt.figure(figsize=(10, 28))
        sns.heatmap(
            heatmap_data_filled,
            cmap=div_cmap,
            center=0,
            cbar_kws={'label': 'Log Fold Change'},
            linewidths=0.5
        )
        plt.title(f"Heatmap of {group} (Filled)")
        plt.savefig(f"{output_folder}heatmap_{group}_rawlogFC.png")
        plt.close()


# In[221]:


def run_go_enrichment(sample_groups, data_dir, output_dir, gene_sets='GO_Biological_Process_2021', organism='Human'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for group_name, selected_samples in sample_groups.items():
        gene_data = []
        
        # Process each sample in the selected group
        for sample in selected_samples:
            file_path = os.path.join(data_dir, sample)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"Processing file: {file_path}")
                
                if 'names' in df.columns and 'logfoldchanges' in df.columns:
                    # Filter genes with positive logfold change
                    filtered_genes = df[df['logfoldchanges'] > 0]['names']
                    if not filtered_genes.empty:
                        gene_data.extend(filtered_genes)
                    else:
                        print(f"No genes passed the filter in {sample}")
                else:
                    print(f"Required columns are missing in {sample}")
            else:
                print(f"File {sample} not found!")
        
        # Remove duplicates
        unique_genes = list(set(gene_data))
        print(f"Unique Genes in {group_name}: {unique_genes}")
        
        if unique_genes:
            # Perform GO enrichment analysis
            results = enrichr(
                gene_list=unique_genes,
                gene_sets=gene_sets,  
                organism=organism
            )

            # Save the enrichment results to a CSV file
            output_file = os.path.join(output_dir, f"{group_name}_go_enrichment_results.csv")
            results.res2d.to_csv(output_file, index=False)
            print(f"Saved enrichment results for {group_name} to {output_file}")

            def style_ax(ax, title_size=60, label_size=45, tick_size=45):
                ax.set_title(ax.get_title(), fontsize=title_size, fontweight='bold')
                ax.set_xlabel(ax.get_xlabel(), fontsize=label_size, fontweight='bold')
                ax.set_ylabel(ax.get_ylabel(), fontsize=label_size, fontweight='bold')
                ax.tick_params(axis='both', which='major', labelsize=tick_size)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight('bold')

           # Barplot
            barplot_output = os.path.join(output_dir, f"{group_name}_go_enrichment_barplot.png")
            ax = barplot(results.res2d, title=f'{group_name} GO Enrichment Analysis', top_term=10, cmap='viridis', figsize=(70, 50))
            style_ax(ax)
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(barplot_output)
            plt.close(fig)

            # Dotplot
            dotplot_output = os.path.join(output_dir, f"{group_name}_go_enrichment_dotplot.png")
            ax = dotplot(results.res2d, title=f'{group_name} GO Enrichment Analysis (Dotplot)', top_term=10, figsize=(70, 50))
            style_ax(ax)
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(dotplot_output)
            plt.close(fig)
            print(f"Saved dotplot for {group_name} to {dotplot_output}")
        else:
            print(f"Gene list for {group_name} is empty. Ensure the input files have valid data.")

# Example of your sample groups
sample_groups = {
    "Group1": ["up100_NN2023_luminal_progenitor_0.csv", "up100_N2021_luminal_progenitor_6.csv"],
    "Group2": ["up100_N2021_luminal_progenitor_1.csv", "up100_R2024_luminal_progenitor_1.csv"],
    "Group3": ["up100_KN23_luminal_progenitor_4.csv", "up100_KK18_luminal_progenitor_1.csv"],
    "Group4": ["up100_N2021_luminal_progenitor_1.csv", "up100_R2024_luminal_progenitor_1.csv", "up100_KN23_luminal_progenitor_4.csv", "up100_KK18_luminal_progenitor_1.csv"],
    "Group5": ["up100_N2021_luminal_progenitor_1.csv", "up100_R2024_luminal_progenitor_1.csv", "up100_KN23_luminal_progenitor_4.csv", "up100_KK18_luminal_progenitor_1.csv", "up100_NN2023_luminal_progenitor_0.csv", "up100_N2021_luminal_progenitor_6.csv"],
    "Group6": ["up100_KN23_luminal_progenitor_1.csv", "up100_JV2021_luminal_progenitor_1.csv"],
    "Group7": ["up100_N2021_luminal_progenitor_3.csv", "up100_R2024_luminal_progenitor_0.csv"],
    "Group8": ["up100_KK18_luminal_progenitor_3.csv", "up100_N2021_luminal_progenitor_3.csv", "up100_R2024_luminal_progenitor_0.csv"],
    "Group9": ["up100_JV2021_luminal_progenitor_2.csv", "up100_NN2023_luminal_progenitor_1.csv"],
    "Group10": ["up100_KK18_luminal_progenitor_0.csv", "up100_KN23_luminal_progenitor_3.csv", "up100_KK18_luminal_progenitor_0.csv", "up100_JV2021_luminal_progenitor_2.csv", "up100_NN2023_luminal_progenitor_1.csv"],
    "Group11": ["up100_KN23_luminal_progenitor_3.csv", "up100_KK18_luminal_progenitor_0.csv", "up100_JV2021_luminal_progenitor_2.csv", "up100_NN2023_luminal_progenitor_1.csv", "up100_KK18_luminal_progenitor_3.csv", "up100_N2021_luminal_progenitor_3.csv", "up100_R2024_luminal_progenitor_0.csv", "up100_KN23_luminal_progenitor_1.csv", "up100_JV2021_luminal_progenitor_1.csv"],
    "Group12": ["up100_N2021_luminal_progenitor_5.csv", "up100_KN23_luminal_progenitor_0.csv"],
    "Group13": ["up100_JV2021_luminal_progenitor_3.csv", "up100_N2021_luminal_progenitor_2.csv"],
    "Group14": ["up100_N2021_luminal_progenitor_7.csv", "up100_KK18_luminal_progenitor_2.csv"],
    "Group15": ["up100_JV2021_luminal_progenitor_3.csv", "up100_N2021_luminal_progenitor_2.csv", "up100_N2021_luminal_progenitor_7.csv", "up100_KK18_luminal_progenitor_2.csv"],
    "Group16": ["up100_JV2021_luminal_progenitor_3.csv", "up100_N2021_luminal_progenitor_2.csv", "up100_N2021_luminal_progenitor_7.csv", "up100_KK18_luminal_progenitor_2.csv", "up100_N2021_luminal_progenitor_5.csv", "up100_KN23_luminal_progenitor_0.csv"],
    "Group17": ["up100_N2021_luminal_progenitor_0.csv", "up100_KN23_luminal_progenitor_2.csv"],
    "Group18": ["up100_KK18_luminal_progenitor_4.csv", "up100_JV2021_luminal_progenitor_4.csv"],
    "Group19": ["up100_R2024_luminal_progenitor_2.csv", "up100_KK18_luminal_progenitor_4.csv", "up100_JV2021_luminal_progenitor_4.csv"],
    "Group20": ["up100_JV2021_luminal_progenitor_3.csv", "up100_N2021_luminal_progenitor_2.csv", "up100_N2021_luminal_progenitor_7.csv", "up100_KK18_luminal_progenitor_2.csv", "up100_N2021_luminal_progenitor_5.csv", "up100_KN23_luminal_progenitor_0.csv", "up100_N2021_luminal_progenitor_0.csv", "up100_KN23_luminal_progenitor_2.csv", "up100_R2024_luminal_progenitor_2.csv", "up100_KK18_luminal_progenitor_4.csv", "up100_JV2021_luminal_progenitor_4.csv"],
    "Group21": ["up100_JV2021_luminal_progenitor_0.csv", "up100_JB2022_luminal_progenitor_2.csv"],
    "Group22": ["up100_N2021_luminal_progenitor_4.csv", "up100_JV2021_luminal_progenitor_0.csv", "up100_JB2022_luminal_progenitor_2.csv"]
}

# Example directories (make sure to adjust these to your file locations)
data_dir = './data/luminal_progenitor/up100/'
output_dir = './data/luminal_progenitor/GO_Enrichment_Results/'

# Run GO enrichment for all groups and save results and plots
run_go_enrichment(sample_groups, data_dir, output_dir)


# In[1]:


def run_go_enrichment_intersect(sample_groups, data_dir, output_dir, gene_sets='GO_Biological_Process_2021', organism='Human'):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from gseapy import enrichr, barplot, dotplot

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for group_name, selected_samples in sample_groups.items():
        gene_lists = []

        # Process each sample in the group
        for sample in selected_samples:
            file_path = os.path.join(data_dir, sample)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"Processing file: {file_path}")

                if 'names' in df.columns and 'logfoldchanges' in df.columns:
                    # Filter genes with positive logfold change
                    filtered_genes = set(df[df['logfoldchanges'] > 0]['names'])
                    if filtered_genes:
                        gene_lists.append(filtered_genes)
                    else:
                        print(f"No genes passed the filter in {sample}")
                else:
                    print(f"Required columns are missing in {sample}")
            else:
                print(f"File {sample} not found!")

        # Take the intersection across all samples in the group
        if gene_lists:
            intersect_genes = set.intersection(*gene_lists)
            print(f"Intersecting Genes in {group_name}: {intersect_genes}")
        else:
            intersect_genes = set()

        if intersect_genes:
            # Perform GO enrichment analysis
            results = enrichr(
                gene_list=list(intersect_genes),
                gene_sets=gene_sets,
                organism=organism
            )

            # Save enrichment results
            output_file = os.path.join(output_dir, f"{group_name}_go_enrichment_results.csv")
            results.res2d.to_csv(output_file, index=False)
            print(f"Saved enrichment results for {group_name} to {output_file}")

            # Helper function to style plots
            def style_ax(ax, title_size=60, label_size=45, tick_size=45):
                ax.set_title(ax.get_title(), fontsize=title_size, fontweight='bold')
                ax.set_xlabel(ax.get_xlabel(), fontsize=label_size, fontweight='bold')
                ax.set_ylabel(ax.get_ylabel(), fontsize=label_size, fontweight='bold')
                ax.tick_params(axis='both', which='major', labelsize=tick_size)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight('bold')

            # Barplot
            barplot_output = os.path.join(output_dir, f"{group_name}_go_enrichment_barplot.png")
            ax = barplot(results.res2d, title=f'{group_name} GO Enrichment Analysis', top_term=10, cmap='viridis', figsize=(70, 50))
            style_ax(ax)
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(barplot_output)
            plt.close(fig)

            # Dotplot
            dotplot_output = os.path.join(output_dir, f"{group_name}_go_enrichment_dotplot.png")
            ax = dotplot(results.res2d, title=f'{group_name} GO Enrichment Analysis (Dotplot)', top_term=10, figsize=(70, 50))
            style_ax(ax)
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(dotplot_output)
            plt.close(fig)
            print(f"Saved dotplot for {group_name} to {dotplot_output}")

        else:
            print(f"No intersecting genes for {group_name}. Skipping GO enrichment.")


# In[2]:


sample_groups = {
    "Group1": ["up100_NN2023_luminal_progenitor_0.csv", "up100_N2021_luminal_progenitor_6.csv"],
    "Group2": ["up100_N2021_luminal_progenitor_1.csv", "up100_R2024_luminal_progenitor_1.csv"],
    "Group3": ["up100_KN23_luminal_progenitor_4.csv", "up100_KK18_luminal_progenitor_1.csv"],
    "Group4": ["up100_N2021_luminal_progenitor_1.csv", "up100_R2024_luminal_progenitor_1.csv", "up100_KN23_luminal_progenitor_4.csv", "up100_KK18_luminal_progenitor_1.csv"],
    "Group5": ["up100_N2021_luminal_progenitor_1.csv", "up100_R2024_luminal_progenitor_1.csv", "up100_KN23_luminal_progenitor_4.csv", "up100_KK18_luminal_progenitor_1.csv", "up100_NN2023_luminal_progenitor_0.csv", "up100_N2021_luminal_progenitor_6.csv"],
    "Group6": ["up100_KN23_luminal_progenitor_1.csv", "up100_JV2021_luminal_progenitor_1.csv"],
    "Group7": ["up100_N2021_luminal_progenitor_3.csv", "up100_R2024_luminal_progenitor_0.csv"],
    "Group8": ["up100_KK18_luminal_progenitor_3.csv", "up100_N2021_luminal_progenitor_3.csv", "up100_R2024_luminal_progenitor_0.csv"],
    "Group9": ["up100_JV2021_luminal_progenitor_2.csv", "up100_NN2023_luminal_progenitor_1.csv"],
    "Group10": ["up100_KK18_luminal_progenitor_0.csv", "up100_KN23_luminal_progenitor_3.csv", "up100_KK18_luminal_progenitor_0.csv", "up100_JV2021_luminal_progenitor_2.csv", "up100_NN2023_luminal_progenitor_1.csv"],
    "Group11": ["up100_KN23_luminal_progenitor_3.csv", "up100_KK18_luminal_progenitor_0.csv", "up100_JV2021_luminal_progenitor_2.csv", "up100_NN2023_luminal_progenitor_1.csv", "up100_KK18_luminal_progenitor_3.csv", "up100_N2021_luminal_progenitor_3.csv", "up100_R2024_luminal_progenitor_0.csv", "up100_KN23_luminal_progenitor_1.csv", "up100_JV2021_luminal_progenitor_1.csv"],
    "Group12": ["up100_N2021_luminal_progenitor_5.csv", "up100_KN23_luminal_progenitor_0.csv"],
    "Group13": ["up100_JV2021_luminal_progenitor_3.csv", "up100_N2021_luminal_progenitor_2.csv"],
    "Group14": ["up100_N2021_luminal_progenitor_7.csv", "up100_KK18_luminal_progenitor_2.csv"],
    "Group15": ["up100_JV2021_luminal_progenitor_3.csv", "up100_N2021_luminal_progenitor_2.csv", "up100_N2021_luminal_progenitor_7.csv", "up100_KK18_luminal_progenitor_2.csv"],
    "Group16": ["up100_JV2021_luminal_progenitor_3.csv", "up100_N2021_luminal_progenitor_2.csv", "up100_N2021_luminal_progenitor_7.csv", "up100_KK18_luminal_progenitor_2.csv", "up100_N2021_luminal_progenitor_5.csv", "up100_KN23_luminal_progenitor_0.csv"],
    "Group17": ["up100_N2021_luminal_progenitor_0.csv", "up100_KN23_luminal_progenitor_2.csv"],
    "Group18": ["up100_KK18_luminal_progenitor_4.csv", "up100_JV2021_luminal_progenitor_4.csv"],
    "Group19": ["up100_R2024_luminal_progenitor_2.csv", "up100_KK18_luminal_progenitor_4.csv", "up100_JV2021_luminal_progenitor_4.csv"],
    "Group20": ["up100_JV2021_luminal_progenitor_3.csv", "up100_N2021_luminal_progenitor_2.csv", "up100_N2021_luminal_progenitor_7.csv", "up100_KK18_luminal_progenitor_2.csv", "up100_N2021_luminal_progenitor_5.csv", "up100_KN23_luminal_progenitor_0.csv", "up100_N2021_luminal_progenitor_0.csv", "up100_KN23_luminal_progenitor_2.csv", "up100_R2024_luminal_progenitor_2.csv", "up100_KK18_luminal_progenitor_4.csv", "up100_JV2021_luminal_progenitor_4.csv"],
    "Group21": ["up100_JV2021_luminal_progenitor_0.csv", "up100_JB2022_luminal_progenitor_2.csv"],
    "Group22": ["up100_N2021_luminal_progenitor_4.csv", "up100_JV2021_luminal_progenitor_0.csv", "up100_JB2022_luminal_progenitor_2.csv"]
}

# Example directories (make sure to adjust these to your file locations)
data_dir = './data/luminal_progenitor/up100/'
output_dir = './data/luminal_progenitor/Intersect_GO_Enrichment_Results/'

# Run GO enrichment for all groups and save results and plots
run_go_enrichment_intersect(sample_groups, data_dir, output_dir)


# In[224]:


sample_groups = {
    "Group1": ["up1000_NN2023_luminal_progenitor_0.csv", "up1000_N2021_luminal_progenitor_6.csv"],
    "Group2": ["up1000_N2021_luminal_progenitor_1.csv", "up1000_R2024_luminal_progenitor_1.csv"],
    "Group3": ["up1000_KN23_luminal_progenitor_4.csv", "up1000_KK18_luminal_progenitor_1.csv"],
    "Group4": ["up1000_N2021_luminal_progenitor_1.csv", "up1000_R2024_luminal_progenitor_1.csv", "up1000_KN23_luminal_progenitor_4.csv", "up1000_KK18_luminal_progenitor_1.csv"],
    "Group5": ["up1000_N2021_luminal_progenitor_1.csv", "up1000_R2024_luminal_progenitor_1.csv", "up1000_KN23_luminal_progenitor_4.csv", "up1000_KK18_luminal_progenitor_1.csv", "up1000_NN2023_luminal_progenitor_0.csv", "up1000_N2021_luminal_progenitor_6.csv"],
    "Group6": ["up1000_KN23_luminal_progenitor_1.csv", "up1000_JV2021_luminal_progenitor_1.csv"],
    "Group7": ["up1000_N2021_luminal_progenitor_3.csv", "up1000_R2024_luminal_progenitor_0.csv"],
    "Group8": ["up1000_KK18_luminal_progenitor_3.csv", "up1000_N2021_luminal_progenitor_3.csv", "up1000_R2024_luminal_progenitor_0.csv"],
    "Group9": ["up1000_JV2021_luminal_progenitor_2.csv", "up1000_NN2023_luminal_progenitor_1.csv"],
    "Group9": ["up1000_JV2021_luminal_progenitor_2.csv", "up1000_NN2023_luminal_progenitor_1.csv"],
    "Group10": ["up1000_KK18_luminal_progenitor_0.csv", "up1000_KN23_luminal_progenitor_3.csv", "up1000_KK18_luminal_progenitor_0.csv", "up1000_JV2021_luminal_progenitor_2.csv", "up1000_NN2023_luminal_progenitor_1.csv"],
    "Group11": ["up1000_KN23_luminal_progenitor_3.csv", "up1000_KK18_luminal_progenitor_0.csv", "up1000_JV2021_luminal_progenitor_2.csv", "up1000_NN2023_luminal_progenitor_1.csv", "up1000_KK18_luminal_progenitor_3.csv", "up1000_N2021_luminal_progenitor_3.csv", "up1000_R2024_luminal_progenitor_0.csv", "up1000_KN23_luminal_progenitor_1.csv", "up1000_JV2021_luminal_progenitor_1.csv"],
    "Group12": ["up1000_N2021_luminal_progenitor_5.csv", "up1000_KN23_luminal_progenitor_0.csv"],
    "Group13": ["up1000_JV2021_luminal_progenitor_3.csv", "up1000_N2021_luminal_progenitor_2.csv"],
    "Group14": ["up1000_N2021_luminal_progenitor_7.csv", "up1000_KK18_luminal_progenitor_2.csv"],
    "Group15": ["up1000_JV2021_luminal_progenitor_3.csv", "up1000_N2021_luminal_progenitor_2.csv", "up1000_N2021_luminal_progenitor_7.csv", "up1000_KK18_luminal_progenitor_2.csv"],
    "Group16": ["up1000_JV2021_luminal_progenitor_3.csv", "up1000_N2021_luminal_progenitor_2.csv", "up1000_N2021_luminal_progenitor_7.csv", "up1000_KK18_luminal_progenitor_2.csv", "up1000_N2021_luminal_progenitor_5.csv", "up1000_KN23_luminal_progenitor_0.csv"],
    "Group17": ["up1000_N2021_luminal_progenitor_0.csv", "up1000_KN23_luminal_progenitor_2.csv"],
    "Group18": ["up1000_KK18_luminal_progenitor_4.csv", "up1000_JV2021_luminal_progenitor_4.csv"],
    "Group19": ["up1000_R2024_luminal_progenitor_2.csv", "up1000_KK18_luminal_progenitor_4.csv", "up1000_JV2021_luminal_progenitor_4.csv"],
    "Group20": ["up1000_JV2021_luminal_progenitor_3.csv", "up1000_N2021_luminal_progenitor_2.csv", "up1000_N2021_luminal_progenitor_7.csv", "up1000_KK18_luminal_progenitor_2.csv", "up1000_N2021_luminal_progenitor_5.csv", "up1000_KN23_luminal_progenitor_0.csv", "up1000_N2021_luminal_progenitor_0.csv", "up1000_KN23_luminal_progenitor_2.csv", "up1000_R2024_luminal_progenitor_2.csv", "up1000_KK18_luminal_progenitor_4.csv", "up1000_JV2021_luminal_progenitor_4.csv"],
    "Group21": ["up1000_JV2021_luminal_progenitor_0.csv", "up1000_JB2022_luminal_progenitor_2.csv"],
    "Group22": ["up1000_N2021_luminal_progenitor_4.csv", "up1000_JV2021_luminal_progenitor_0.csv", "up1000_JB2022_luminal_progenitor_2.csv"]
}

# Example directories (make sure to adjust these to your file locations)
data_dir = './data/luminal_progenitor/up1000/'
output_dir = './data/luminal_progenitor/1000_Intersect_GO_Enrichment_Results/'

# Run GO enrichment for all groups and save results and plots
run_go_enrichment_intersect(sample_groups, data_dir, output_dir)


# # SCVI

# In[111]:


#reload filtered out immune
filtered_out_immune = sc.read('luminal_basal_subset.h5ad')


# In[112]:


#new model, copy of immune cell removed dataset called alldata
alldata = filtered_out_immune.copy(filtered_out_immune)


# In[113]:


sc.pp.calculate_qc_metrics(alldata, inplace=True)


# In[114]:


scvi.model.SCVI.setup_anndata(alldata, layer = "counts",
                            categorical_covariate_keys=["Experiment", "Sample", "Batch"],
                     continuous_covariate_keys=['pct_counts_mt', 'total_counts', 'pct_counts_ribo'])

model = scvi.model.SCVI(alldata)


# In[115]:


import time

t1 = time.time()
model.train()
print(time.time() - t1)


# In[116]:


alldata.obsm['X_scVI'] = model.get_latent_representation()
alldata.layers['scvi_normalized'] = model.get_normalized_expression(library_size = 1e4)

sc.pp.neighbors(alldata, use_rep = 'X_scVI')
sc.tl.umap(alldata)


# In[118]:


sc.tl.leiden(alldata, resolution = 0.5)


# In[126]:


sc.pl.umap(alldata, color = ['Batch', 'leiden', 'Sample'], ncols=3) 


# In[120]:


sc.pl.umap(alldata, color = ['leiden'], frameon = False, legend_loc = "on data")


# In[122]:


markers_scvi = model.differential_expression(groupby = 'leiden', group1=['0', '1', '3', '4', '5','6','7','8','9','10','11'])
markers_scvi = markers_scvi[(markers_scvi['is_de_fdr_0.05']) & (markers_scvi.lfc_mean > .5)]
markers_scvi


# In[ ]:


markers_scvi.loc[markers_scvi.group1 == '3', ['lfc_mean', 'raw_mean1', 'raw_mean2']].sort_values(by=['lfc_mean'], inplace=False, ascending=False)


# In[ ]:


#EPCAM & KRT19 & ITGA6 (CD49F)#
sc.pl.umap(alldata, color=[ 'KRT19', 'ITGA6'], ncols=3)


# In[123]:


#BASAL Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(alldata, color=['ACTA2', 'TAGLN', 'MYL9', 'TPM2', 'ACTG2', 'KRT5', 'ITGA6', 'KRT14', 'KRT17', 'CCND2', 'DKK3', 'SPARC', 'VIM'], ncols=4)


# In[143]:


markers_scvi.loc['TAGLN', ['proba_de', 'group1']]


# In[124]:


#LUMINAL PROGENITOR Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(alldata, color=['LTF', 'SLPI', 'RARRES1', 'ALDH1A3', 'KRT15', 'PI3', 'S100A9', 'CD24', 'KIT', 'ALDH1A3', 'KRT15', 'PI3', 'S100A9', 'CD24', 'KIT'], ncols=4)


# In[136]:


markers_scvi.loc['KRT15', ['proba_de', 'group1']]


# In[125]:


#MATURE LUMINAL Markers for Epithelial Breast Subtypes: Bartlett et al. 2021#
sc.pl.umap(alldata, color=['AREG','PRLR', 'PGR', 'ESR1', 'FOXA1', 'TTF1', 'TFF3', 'SYTL2','IGF1R','AKT1'], ncols=4)


# In[142]:


markers_scvi.loc['TFF3', ['proba_de', 'group1']]


# In[ ]:


sc.pl.umap(alldata, color = [ 'CD14', 'CD83'], vmax = 3)


# In[144]:


cell_type = {"0":"Basal",
"1":"Basal",
"2":"Luminal Mature",
"3":"Luminal Progenitor",
"4":"Luminal Mature",
"5":"Basal",
"6":"Luminal Progenitor",
"7":"Basal",
"8":"Luminal Mature",
"9":"Luminal Progenitor",
"10":"Basal",
"11":"Basal"}


# In[145]:


alldata.obs['cell type'] = alldata.obs.leiden.map(cell_type)


# In[147]:


sc.pl.umap(alldata, color = ['cell type'], frameon = False)


# In[151]:


sc.pl.umap(alldata, color = ['cell type', 'Batch'], ncols=1)


# In[148]:


#save annotated anndata object for later:
alldata.write("./data/29_06_2025_alldata_scvi_epithelial_annotated.h5ad")


# In[34]:


#reload for later
alldata = sc.read_h5ad("./data/29_06_2025_alldata_scvi_epithelial_annotated.h5ad")
print(alldata)
print(alldata.obs['cell type'].head())
sc.pl.umap(alldata, color=['cell type'])


# In[35]:


if 'Batch' in alldata.obs.columns:
    counts_per_study = alldata.obs.groupby('Batch')['cell type'].value_counts().unstack(fill_value=0)
    counts_per_study_of_interest = counts_per_study[cell_types_of_interest]
    perc_per_study = counts_per_study_of_interest.div(counts_per_study_of_interest.sum(axis=1), axis=0) * 100
    counts_per_study_of_interest.to_csv("cell_counts_per_study_scvi.csv")
    perc_per_study.to_csv("cell_percentages_per_study_scvi.csv")


# In[4]:


scvi_basal = alldata[alldata.obs['cell type'].isin(['Basal'])]
sc.pl.umap(scvi_basal, color = ['cell type'], frameon = False)


# In[5]:


scvi_mature_luminal = alldata[alldata.obs['cell type'].isin(['Luminal Mature'])]
sc.pl.umap(scvi_mature_luminal, color = ['cell type'], frameon = False)


# In[6]:


scvi_luminal_progenitor = alldata[alldata.obs['cell type'].isin(['Luminal Progenitor'])]
sc.pl.umap(scvi_luminal_progenitor, color = ['cell type'], frameon = False)


# ### Basal SCVI ###

# In[7]:


sc.pp.highly_variable_genes(scvi_basal, n_top_genes = 2000)
scvi_basal_hv = scvi_basal[:, scvi_basal.var['highly_variable']].copy()
sc.pp.pca(scvi_basal_hv)
sc.pp.neighbors(scvi_basal_hv, n_pcs = 20)
sc.tl.leiden(scvi_basal_hv, resolution = 0.4)
sc.tl.umap(scvi_basal_hv)
sc.pl.umap(scvi_basal_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[8]:


sc.pl.umap(scvi_basal_hv, color=['ACTA2', 'TAGLN', 'MYL9', 'TPM2', 'ACTG2', 'KRT5', 'ITGA6', 'KRT14', 'KRT17', 'CCND2', 'DKK3', 'SPARC'], ncols=4)


# In[19]:


save_top_marker_genes(scvi_basal_hv, 'scvi', 'basal')


# In[9]:


sc.tl.rank_genes_groups(scvi_basal_hv, 'leiden')
sc.pl.rank_genes_groups(scvi_basal_hv, n_genes=20, sharey=False)


# In[47]:


sc.pl.umap(scvi_basal_hv, color = ['leiden'], frameon = False, legend_loc = "on data")


# ### LM SCVI

# In[21]:


sc.pp.highly_variable_genes(scvi_mature_luminal, n_top_genes = 2000)
scvi_mature_luminal_hv = scvi_mature_luminal[:, scvi_mature_luminal.var['highly_variable']].copy()
sc.pp.pca(scvi_mature_luminal_hv)
sc.pp.neighbors(scvi_mature_luminal_hv, n_pcs = 20)
sc.tl.leiden(scvi_mature_luminal_hv, resolution = 0.45)
sc.tl.umap(scvi_mature_luminal_hv)
sc.pl.umap(scvi_mature_luminal_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[48]:


sc.pl.umap(scvi_mature_luminal_hv, color = ['leiden'], frameon = False, legend_loc = "on data")


# In[22]:


sc.pl.umap(scvi_mature_luminal_hv, color=['AREG','PRLR', 'PGR', 'ESR1', 'FOXA1', 'TTF1', 'TFF3', 'SYTL2','IGF1R','AKT1'], ncols=4)


# In[26]:


save_top_marker_genes(scvi_mature_luminal_hv, 'scvi', 'mature_luminal')


# In[23]:


sc.tl.rank_genes_groups(scvi_mature_luminal_hv, 'leiden')
sc.pl.rank_genes_groups(scvi_mature_luminal_hv, n_genes=20, sharey=False)


# ### LP SCVI 

# In[24]:


sc.pp.highly_variable_genes(scvi_luminal_progenitor, n_top_genes = 2000)
scvi_luminal_progenitor_hv = scvi_luminal_progenitor[:, scvi_luminal_progenitor.var['highly_variable']].copy()
sc.pp.pca(scvi_luminal_progenitor_hv)
sc.pp.neighbors(scvi_luminal_progenitor_hv, n_pcs = 20)
sc.tl.leiden(scvi_luminal_progenitor_hv, resolution = 0.5)
sc.tl.umap(scvi_luminal_progenitor_hv)
sc.pl.umap(scvi_luminal_progenitor_hv, color=['leiden', 'cell type', 'Batch', 'Sample'], ncols=2)


# In[50]:


sc.pl.umap(scvi_luminal_progenitor_hv, color = ['leiden'], frameon = False, legend_loc = "on data")


# In[15]:


sc.pl.umap(scvi_luminal_progenitor_hv, color=['LTF', 'SLPI', 'RARRES1', 'ALDH1A3', 'KRT15', 'PI3', 'S100A9', 'CD24', 'KIT', 'ALDH1A3', 'KRT15', 'PI3', 'S100A9', 'CD24', 'KIT'], ncols=4)


# In[27]:


save_top_marker_genes(scvi_luminal_progenitor_hv, 'scvi', 'luminal_progenitor')


# In[25]:


sc.tl.rank_genes_groups(scvi_luminal_progenitor_hv, 'leiden')
sc.pl.rank_genes_groups(scvi_luminal_progenitor_hv, n_genes=20, sharey=False)


# ### Dendrograms WITH SCVI

# In[41]:


build_rank_dendrogram('./data/basal/up30/', top_n=30, title="Basal subcluster similarity w/ SCVI (top 30)")


# In[38]:


build_rank_dendrogram('./data/basal/up100/', top_n=100, title="Basal subcluster similarity w/ SCVI (top 100)")


# In[39]:


build_rank_dendrogram('./data/basal/up1000/', top_n=1000, title="Basal subcluster similarity w/ SCVI (top 1000)")


# In[44]:


build_rank_dendrogram('./data/mature_luminal/up30/', top_n=30, title="Luminal Mature subcluster similarity w/ SCVI (top 30)")


# In[43]:


build_rank_dendrogram('./data/mature_luminal/up1000/', top_n=1000, title="Luminal Mature subcluster similarity w/ SCVI (top 1000)")


# In[42]:


build_rank_dendrogram('./data/mature_luminal/up100/', top_n=100, title="Luminal Mature subcluster similarity w/ SCVI (top 100)")


# In[37]:


build_rank_dendrogram('./data/luminal_progenitor/up30/', top_n=30, title="Luminal Progenitor subcluster similarity w/ SCVI (top 30)")


# In[40]:


build_rank_dendrogram('./data/luminal_progenitor/up1000/', top_n=1000, title="Luminal Progenitor subcluster similarity w/ SCVI (top 1000)")


# In[34]:


build_rank_dendrogram('./data/luminal_progenitor/up100/', top_n=100, title="Luminal Progenitor subcluster similarity w/ SCVI (top 100)")


# In[ ]:


build_rank_dendrogram('./data/All_epithelial/Up100/', top_n=100, title="All Epithelial subcluster similarity (top 100)")

