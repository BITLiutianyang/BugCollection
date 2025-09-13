import os
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import shap
from featuretools.primitives import TransformPrimitive
# from featuretools.variable_types import Numeric
import featuretools as ft
import joblib
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
import lizard
import math
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import plot_tree

from sklearn.neural_network import MLPClassifier  # 添加 MLPClassifier

import pickle

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from transformers import RobertaTokenizer, RobertaModel

import featuretools as ft
from featuretools.primitives import MultiplyNumeric

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import matplotlib.font_manager as fm

from logitboost import LogitBoost 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 加载数据
def load_data(input_base_path):
    data_dict = {
        "index": [],
        "description": [],
        "code_diff": [],
        "flag": [],
        "label": [],
        
        "file_content_before": [],
        
        "adds": [],
        "deletes": [],
        'semgrep_result': [],
        'llm_gpt_result': [],
        'llm_gpt_bug_result': []
    }
    
    input_path = os.path.join(input_base_path, "all_data_semgrep_description_llmpatch.jsonl")  
    with open(input_path, 'r') as rf:
        lines = rf.readlines()
        for line in lines:
            data = json.loads(line)
            index = data['idx']
            flag = data["flag"]
            if flag == "NDS":
                label = 1
            else:
                label = 0
            
            adds = data['stats']['additions']
            deletes = data['stats']['deletions']
            
            description = data["description_pro_new_filter"].replace("【What】","").replace("【Why】","").strip()
            
            code_diff = data["files"][0]['patch']
        
            
            semgrep_before = data['files'][0]['semgrep_result_before_onemore'][0]
            semgrep_after = data['files'][0]['semgrep_result_after_onemore'][0]
            if semgrep_before is True and semgrep_after is False:
                semgrep_result = 0.9
            elif semgrep_before is True and semgrep_after is True:
                semgrep_result = 0.6
            else:
                semgrep_result = 0.3
            
            if isinstance(data['files'][0]['llm-gpt-4o-before'], dict):
                llm_gpt_before = data['files'][0]['llm-gpt-4o-before']['answer']
            elif isinstance(data['files'][0]['llm-gpt-4o-before'], list):
                llm_gpt_before = data['files'][0]['llm-gpt-4o-before'][0]['answer']
            if isinstance(data['files'][0]['llm-gpt-4o-after'], dict):
                llm_gpt_after = data['files'][0]['llm-gpt-4o-after']['answer']
            elif isinstance(data['files'][0]['llm-gpt-4o-after'], list):
                llm_gpt_after = data['files'][0]['llm-gpt-4o-after'][0]['answer']
            
            if llm_gpt_before == 'yes' and llm_gpt_after == 'no':
                llm_gpt_result = 0.9
            elif llm_gpt_before == 'yes' and llm_gpt_after == 'yes':
                llm_gpt_result = 0.6
            else:
                llm_gpt_result = 0.3
            
            llm_gpt_bug = data['files'][0]['llm_path_with_description_pro_new'][0]
            if llm_gpt_bug == 'yes':
                llm_gpt_bug_result = 1
            else:
                llm_gpt_bug_result = 0
            file_content_before = data['files'][0]['old_content']
            data_dict['index'].append(index)
            data_dict['description'].append(description)
            data_dict['code_diff'].append(code_diff)
            data_dict['flag'].append(flag)
            data_dict['label'].append(label)
            data_dict['adds'].append(adds)
            data_dict['deletes'].append(deletes)
            data_dict['file_content_before'].append(file_content_before)
            data_dict['semgrep_result'].append(semgrep_result)
            data_dict['llm_gpt_result'].append(llm_gpt_result)
            data_dict['llm_gpt_bug_result'].append(llm_gpt_bug_result)  
    return pd.DataFrame(data_dict)
def extract_embeddings(texts, flag="tfidf"):
    max_features = 512
    if flag == "tfidf":
        vectorizer = TfidfVectorizer(max_features=max_features)
        embeddings = vectorizer.fit_transform(texts.tolist()).toarray()
    elif flag == "bow":
        vectorizer = CountVectorizer(max_features=max_features)
        embeddings = vectorizer.fit_transform(texts.tolist()).toarray()
    elif flag == "bert":
        model_path = "path_to_model"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        
        # 确保 texts 是一个字符串列表
        texts = texts.tolist()  # 如果 texts 是 pandas Series
        texts = [str(text) for text in texts]  # 将每个元素转换为字符串
        
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # Get the [CLS] token embedding
        # Reduce dimensionality using PCA
        pca = PCA(n_components=max_features)
        embeddings = pca.fit_transform(cls_embeddings)
    else:
        raise ValueError(f"Unsupported embedding method: {flag}")  
    return np.array(embeddings)

def reduce_embeddings(embeddings, n_components=32):
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

def calculate_code_complexity(code_content):
    analysis_result = lizard.analyze_file.analyze_source_code("temp.java", code_content)
    result_vector = {
        "lines_of_code": analysis_result.nloc,
        "token_num": analysis_result.token_count,
        "complexity": analysis_result.CCN,
        "function_num": len(analysis_result.function_list),
    }
    return result_vector
def extract_additional_features(df):
    keyword_list = ["bug", "fix", "issue", "error"]
    counts = df['description'].str.lower().apply(lambda desc: sum(desc.count(keyword) for keyword in keyword_list))
    df['keyword_count'] = counts
    return df

# 特征提取与降维
def extract_features(df, flag = "bert", output_dir = 'path_to_ouput'):
    output_file = os.path.join(output_dir, f"features_{flag}.pkl")
    if os.path.exists(output_file):
        print(f"Loading existing features from {output_file}")
        df = pd.read_pickle(output_file)
    else:
        print("Extracting features and saving to file...")
        discrip_embeddings = extract_embeddings(texts=df['description'], flag=flag)
        df['description_embedding'] = discrip_embeddings.tolist()
        java_embeddings = extract_embeddings(texts=df['code_diff'], flag=flag)
        df['java_embedding'] = java_embeddings.tolist()

        complexity_df = df['file_content_before'].apply(calculate_code_complexity).apply(pd.Series)
        df = pd.concat([df, complexity_df], axis=1)
        
        df = extract_additional_features(df)
        df.drop(columns=['description', 'code_diff', "flag"], inplace=True)
        df.to_pickle(output_file)
        print(f"Features saved to {output_file}")
    
    columns = df.columns
    print(columns)
    
    return df
def tune_xgboost(X_train, y_train):
    # 验证数据格式
    y_train = np.array(y_train)
    assert isinstance(X_train, np.ndarray), "X_train 必须是 NumPy 数组"
    assert isinstance(y_train, np.ndarray), "y_train 必须是 NumPy 数组"
    
    
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'n_estimators': [100, 200, 300],
        'gamma': [0, 0.1, 0.2]
    } 
    tscv = TimeSeriesSplit(n_splits=5)
    xgb_model = xgb.XGBClassifier(eval_metric='logloss')
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                               cv=tscv, scoring='accuracy', n_jobs=-1, verbose=2)
    try:
        grid_search.fit(X_train, y_train)
    except OSError as e:
        print(f"捕获到 OSError: {e}")
        # 进一步处理或记录日志

    return grid_search.best_estimator_

def tune_random_forest(X_train, y_train, model):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    print(f"Best parameters found for random forest: {grid_search.best_params_}")
    return grid_search.best_estimator_
    
def tune_mlp(X_train, y_train):
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
    }
    mlp = MLPClassifier(max_iter=1000, random_state=42)
    grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_  
    
def scale_features(X_train, X_test, method='standard'):
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaling method. Choose 'standard' or 'minmax'.")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def draw_importance(model, feature_names, name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        total_features = len(importances)

        # --- 分组定义 ---
        # Group1 是前10个特征的平均重要性
        group1_indices = np.arange(0, 10)
        group1_importance = importances[group1_indices]
        group1_feature_names = [feature_names[i] for i in group1_indices]
        print(group1_feature_names)

        # Group2 是随后512个特征的平均重要性
        group2_indices = np.arange(10, 522)
        group2_importance = np.mean(importances[group2_indices]) if len(group2_indices) > 0 else 0

        # Group3 是再随后的每个特征
        group3_indices = np.arange(522, total_features)
        group3_importances = np.mean(importances[group3_indices]) if len(group3_indices) > 0 else 0
        

        # --- 合并特征 ---
        combined_importances =  list(group1_importance) + [group2_importance, group3_importances]
        combined_feature_names = group1_feature_names + ['description', 'code diff']
        
        # --- 过滤 importance 分数为 0 的特征 ---
        non_zero_indices = [i for i, importance in enumerate(combined_importances) if importance > 0]
        filtered_importances = [combined_importances[i] for i in non_zero_indices]
        filtered_feature_names = [combined_feature_names[i] for i in non_zero_indices]

        # --- 排序 ---
        # 将特征名称和重要性组合成一个列表
        combined_features = list(zip(filtered_feature_names, filtered_importances))
        # 按重要性降序排序
        sorted_features = sorted(combined_features, key=lambda x: x[1], reverse=True)
        # 分离排序后的特征名称和重要性
        sorted_feature_names, sorted_importances = zip(*sorted_features)

        # --- 绘制重要性图 ---
        plt.figure(figsize=(8, 3))
        # 设置字体为 Times New Roman
        font_path = "/app/bugcollection/font/times.ttf"
        font_prop = fm.FontProperties(fname=font_path)
        fm.fontManager.addfont(font_path)  # This is the key step that might be missing
        # Set the font family
        plt.rcParams['font.family'] = font_prop.get_name()
        
        
        ax = sns.barplot(x=sorted_importances, y=sorted_feature_names, palette="viridis")
        
         # 设置坐标轴刻度字号
        ax.tick_params(axis='both', labelsize=14)
        
        # for i, value in enumerate(sorted_importances):
        #     plt.text(value + 0.001, i, f"{value:.4f}", va='center', fontsize=12)
                # 添加数据标签并确保不超出范围
        max_importance = max(sorted_importances)
        for i, value in enumerate(sorted_importances):
            # 动态调整标签位置：如果值太大就放在柱子内部
            if value > max_importance * 0.8:  # 如果值接近最大值
                text_x = value * 0.95  # 放在柱子内部
                color = 'white'        # 白色文字
                ha = 'right'           # 右对齐
            else:
                text_x = value + max_importance * 0.02  # 放在柱子右侧
                color = 'black'        # 黑色文字
                ha = 'left'            # 左对齐
            
            ax.text(text_x, i, f"{value:.4f}", 
                   va='center', ha=ha, 
                   color=color, fontsize=14)
            
        # 调整x轴范围，留出标签空间
        ax.set_xlim(0, max_importance * 1.15)
            
        # 设置标题和轴标签
        # plt.title("Feature Importance", fontsize=14, fontfamily='serif')
        plt.xlabel("Importance", fontsize=14, fontfamily='serif')
        plt.ylabel("Features", fontsize=14, fontfamily='serif')
        plt.tight_layout()
        plt.savefig(os.path.join("path_to_visualizations", f"{name}_importance.png"), dpi=300, bbox_inches='tight')
        plt.close()
# 示例用法
# 假设 model 已训练完成，feature_names 是特征名称列表，name 是模型名称
# draw_importance(model, feature_names, name)
def train_and_evaluate_models(X, y, model_dir):
    X_train_t, X_test_t, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # original_X_test_indices = X_test.index
    X_train = X_train_t.drop(columns=['index'])
    X_test = X_test_t.drop(columns=['index'])
    
    # column_names = X_train.columns.tolist()
    # print("X_train 的列名列表：", column_names)
    
    # 获取y_test中正负样本的数量
    positive_count = (y_test == 1).sum()
    negative_count = (y_test == 0).sum()
    
    print(f"Number of positive samples in y_test: {positive_count}")
    print(f"Number of negative samples in y_test: {negative_count}")
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    models = {
        'Logistic_Regression': LogisticRegression(solver='lbfgs', max_iter=1000),
        'Decision_Tree': DecisionTreeClassifier(random_state=42),
        'Random_Forest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(eval_metric='logloss'),
        'LogitBoost': LogitBoost(n_estimators=100, learning_rate=0.1),
        "Gradient_Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    }
  
    results = {}
    print(X_train.head())
    X_train =X_train.astype(np.float64) 
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    poly = PolynomialFeatures(degree=1)
    X_train_scaled = poly.fit_transform(X_train_scaled)
    X_test_scaled = poly.transform(X_test_scaled)
    
    # print(X_train_scaled)
    
    # 保存缩放器
    scaler_path = os.path.join(models_path, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
  
    feature_names = X_train.columns.tolist()  # 获取特征名称
    if hasattr(poly, 'get_feature_names_out'):
        feature_names = poly.get_feature_names_out(feature_names)
    else:
        feature_names = [f"poly_{i}" for i in range(X_train_scaled.shape[1])]
    
    # column_names = X_train.columns.tolist()
    # X_train_scaled_df = pd.DataFrame(X_train_scaled)
    # column_names = X_train_scaled_df.columns.tolist()
    # print("X_train_scaled_df 的列名列表：", column_names)
  
    for name, model in models.items():
        model_filename = os.path.join(model_dir, f"{name}_model.pkl")
        if os.path.exists(model_filename):
            with open(model_filename, 'rb') as f:
                model = pickle.load(f)
                print(f"加载已存在的模型: {name}")
        else:
         
            print(f"Training model: {name}")
            
            if name == 'XGBoost':
                model = tune_xgboost(X_train_scaled, y_train)
                
            if name == "Random_Forest":
                model = tune_random_forest(X_train_scaled, y_train, model)
            
            if name == "MLP":
                model = tune_mlp(X_train_scaled, y_train)
            
            model.fit(X_train_scaled, y_train)
            
            # 保存模型
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)
            
            score = model.score(X_test_scaled, y_test)
            print(f"{name}: {score}")
                # 绘制特征重要性图
        draw_importance(model, feature_names, name)
        if name == "Random_Forest":
            # 假设 model 是训练好的随机森林模型
            tree_index = 0  # 选择第 0 棵树
            tree = model.estimators_[tree_index]
            # 可视化该树的结构
            plt.figure(figsize=(20, 10))
            plot_tree(tree, feature_names=feature_names, filled=True, rounded=True, proportion=True)
            plt.title(f"Tree {tree_index} Structure")
            output_path = os.path.join("/app/bugcollection/implement/filtering_patches/visualizations", f"{name}_single_tree.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            
        # 进行预测
        y_pred = model.predict(X_test_scaled)
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # 假阳率
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # 假阴率
        
        # 存储评估指标
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr,
            'fnr': fnr,
            'y_pred': y_pred
        }
  
  
    return results, X_train_scaled, X_test_scaled, y_train, y_test, X_test_t

# 保存测试结果
def save_results(df_or, original_data, predictions, output_dir, model_name, flag):
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  output_path = os.path.join(output_dir, f"{model_name}_predictions_{flag}.csv")
  file_exists = os.path.exists(output_path)

  # 将预测结果添加到相应的行中 
  # for idx, pred in predictions:
  for idx in range(len(predictions)):
      pred = predictions[idx]
      data = original_data.iloc[idx]
      index = data['index']
      
      row = df_or[df_or['index'] == index].copy()
      
      row['prediction'] = int(pred)
      # data_dict = json.dumps(row.to_dict()) + '\n'
      # 将 row 转换为字符串格式并写入文件
      with open(output_path, "a", encoding="utf-8") as wf:
          if not file_exists:
              row.to_csv(wf, header=True, index=False)
              file_exists = True  # 标记文件已存在，后续不再写入表头
          else:
              row.to_csv(wf, header=False, index=False)

# 将嵌入向量展开为单独的特征列
def expand_embeddings(df, column_name):
    # 将嵌入向量转换为 DataFrame，每维一个列
    embeddings_df = pd.DataFrame(df[column_name].tolist(), index=df.index)
    # 重命名列以避免重复
    embeddings_df.columns = [f"{column_name}_{i}" for i in embeddings_df.columns]
    return embeddings_df
# 主函数
def main(input_path, output_dir, models_path):
    root_path = ""
    flag = "tfidf"
    # 加载数据
    df_or = load_data(input_path)
    
    # 提取特征
    df = extract_features(df_or, flag)
    
    # 分离特征和标签
    # X = df.drop(columns=['label', 'index', "file_content_before"])
    X = df.drop(columns=['label', "file_content_before"])
    xcolumns = X.columns
    print("xcolumns", xcolumns)
    
    y = df['label']
    
    # 展开所有嵌入列
    X_expanded = X.drop(columns=['description_embedding', 'java_embedding'])
    X_expanded = pd.concat([
        X_expanded,
        expand_embeddings(X, 'description_embedding'),
        expand_embeddings(X, 'java_embedding')
    ], axis=1)
    
    X_expanded = X_expanded.fillna(method='ffill')  # 用前一个非缺失值填充
    print("X_expanded", X_expanded)
    print(X_expanded.columns)
    
    # 训练和评估模型
    results, X_train_scaled, X_test_scaled, y_train, y_test, X_test_t = train_and_evaluate_models(X_expanded, y, models_path)
    
    # 保存训练集和测试集的嵌入及对应的标签
    train_data = {'embeddings': X_train_scaled, 'labels': y_train}
    test_data = {'embeddings': X_test_scaled, 'labels': y_test}
    train_embeddings_path = os.path.join(root_path, f'train_embeddings_{flag}.pkl')
    test_embeddings_path = os.path.join(root_path, f'test_embeddings_{flag}.pkl')
    with open(train_embeddings_path, 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(test_embeddings_path, 'wb') as f:
        pickle.dump(test_data, f)
    
    # 打印评估结果
    for name, metrics in results.items():
        print(f'Model: {name}')
        for metric, value in metrics.items():
            if isinstance(value, np.ndarray):
                continue
            print(f'  {metric}: {value:.4f}')
        print()
      
  
    # # # 保存测试结果
    save_results(df_or, X_test_t, results['Logistic_Regression']['y_pred'], output_dir, 'logistic_regression', flag)
    save_results(df_or, X_test_t, results['Decision_Tree']['y_pred'], output_dir, 'decision_tree', flag)
    save_results(df_or, X_test_t, results['Random_Forest']['y_pred'], output_dir, 'random_forest', flag)
    save_results(df_or, X_test_t, results['XGBoost']['y_pred'], output_dir, 'xgboost', flag)
    save_results(df_or, X_test_t, results['LogitBoost']['y_pred'], output_dir, 'logiboost', flag)
    save_results(df_or, X_test_t, results['Gradient_Boosting']['y_pred'], output_dir, 'gradient_boosting', flag)
    save_results(df_or, X_test_t, results['MLP']['y_pred'], output_dir, 'mlp', flag)
      

if __name__ == "__main__":
    input_path = ''
    output_dir = ''
    models_path = ''
    main(input_path, output_dir, models_path)# 
