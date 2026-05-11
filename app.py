import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request, redirect, url_for, send_file, session
from werkzeug.utils import secure_filename
import io
import uuid
import re

# 初始化Flask应用
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # 用于session加密
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 限制10MB

# 创建上传目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 安全的文件名处理（保留中文）
def safe_filename(filename):
    # 只移除危险字符，保留中文、数字、字母和下划线
    filename = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9_.-]', '_', filename)
    # 确保文件名不为空
    if not filename:
        filename = 'upload'
    return filename

# 辅助函数：从session中读取数据
def get_data():
    if 'file_path' not in session:
        return None
    
    file_path = session['file_path']
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    
    try:
        if session['filename'].endswith('.csv'):
            # 尝试多种编码读取CSV文件
            encodings = ['utf-8-sig', 'gbk', 'gb2312', 'cp1252']
            for encoding in encodings:
                try:
                    return pd.read_csv(file_path, encoding=encoding)
                except UnicodeDecodeError:
                    continue
            # 如果所有编码都失败，尝试不指定编码
            return pd.read_csv(file_path, engine='python')
        else:
            # Excel文件使用openpyxl引擎，添加更多错误处理
            return pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        print(f"读取文件失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 辅助函数：保存数据到文件
def save_data(df):
    if 'file_path' not in session:
        return False
    
    try:
        if session['filename'].endswith('.csv'):
            df.to_csv(session['file_path'], index=False, encoding='utf-8-sig')
        else:
            df.to_excel(session['file_path'], index=False, engine='openpyxl')
        return True
    except Exception as e:
        print(f"保存文件失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# 首页 - 文件上传
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 检查是否有文件上传
        if 'file' not in request.files:
            return render_template('index.html', error='没有选择文件')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='没有选择文件')
        
        if file and allowed_file(file.filename):
            # 生成唯一文件名，保留中文
            original_filename = file.filename
            file_ext = original_filename.rsplit('.', 1)[1].lower()
            unique_id = str(uuid.uuid4())
            # 只使用UUID作为文件名，彻底避免中文和特殊字符问题
            filename = f"{unique_id}.{file_ext}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            print(f"文件已保存: {file_path}")
            print(f"原始文件名: {original_filename}")
            
            # 验证文件是否能正常读取
            try:
                if file_ext == 'csv':
                    # 尝试多种编码读取CSV文件
                    encodings = ['utf-8-sig', 'gbk', 'gb2312', 'cp1252']
                    df = None
                    for encoding in encodings:
                        try:
                            df = pd.read_csv(file_path, encoding=encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    if df is None:
                        df = pd.read_csv(file_path, engine='python')
                else:
                    df = pd.read_excel(file_path, engine='openpyxl')
                
                print(f"文件读取成功，数据形状: {df.shape}")
                
                # 在session中存储文件路径和原始文件名
                session['filename'] = original_filename
                session['file_path'] = file_path
                
                return redirect(url_for('data_preview'))
            except Exception as e:
                print(f"文件验证失败: {str(e)}")
                import traceback
                traceback.print_exc()
                
                os.remove(file_path)
                return render_template('index.html', error=f'文件读取失败: {str(e)}')
        
        return render_template('index.html', error='不支持的文件格式，请上传CSV或Excel文件')
    
    return render_template('index.html')

# 数据预览页面
@app.route('/data-preview')
def data_preview():
    df = get_data()
    if df is None:
        return redirect(url_for('index'))
    
    # 获取数据基本信息
    info = {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'columns_list': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'total_missing': df.isnull().sum().sum()
    }
    
    # 预览前10行数据
    preview_data = df.head(10).to_html(classes='table table-striped table-hover', index=False)
    
    return render_template('data_preview.html', 
                          filename=session['filename'],
                          info=info,
                          preview_data=preview_data)

# 数据清洗页面
@app.route('/data-clean', methods=['GET', 'POST'])
def data_clean():
    df = get_data()
    if df is None:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        action = request.form.get('action')
        column = request.form.get('column')
        
        if action == 'drop_missing':
            # 删除缺失值
            df = df.dropna()
        elif action == 'fill_mean':
            # 用均值填充数值型列
            if column and pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].fillna(df[column].mean())
        elif action == 'fill_median':
            # 用中位数填充数值型列
            if column and pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].fillna(df[column].median())
        elif action == 'fill_mode':
            # 用众数填充
            if column:
                df[column] = df[column].fillna(df[column].mode()[0])
        elif action == 'detect_outliers':
            # 检测异常值（IQR方法）
            if column and pd.api.types.is_numeric_dtype(df[column]):
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                outlier_count = len(outliers)
                session['outlier_info'] = {
                    'column': column,
                    'count': outlier_count,
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2)
                }
        elif action == 'remove_outliers':
            # 移除异常值
            if column and pd.api.types.is_numeric_dtype(df[column]):
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
        # 保存修改后的数据
        save_data(df)
    
    # 获取更新后的数据信息
    info = {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'missing_values': df.isnull().sum().to_dict(),
        'total_missing': df.isnull().sum().sum(),
        'numeric_columns': df.select_dtypes(include=['number']).columns.tolist()
    }
    
    outlier_info = session.pop('outlier_info', None)
    
    return render_template('data_clean.html',
                          filename=session['filename'],
                          info=info,
                          outlier_info=outlier_info)

# 可视化页面
# 可视化页面
@app.route('/visualization', methods=['GET', 'POST'])
def visualization():
    df = get_data()
    if df is None:
        return redirect(url_for('index'))
    
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    plot_html = None
    error = None
    
    if request.method == 'POST':
        try:
            chart_type = request.form.get('chart_type')
            x_col = request.form.get('x_column')
            y_col = request.form.get('y_column')
            color_col = request.form.get('color_column')
            
            # 构建图表参数字典
            fig_kwargs = {
                'title': '',
                'template': 'plotly_white'
            }
            
            # 只有当颜色列不为空时才添加color参数
            if color_col and color_col.strip() != '':
                fig_kwargs['color'] = color_col
            
            if chart_type == 'scatter' and x_col and y_col:
                fig_kwargs['title'] = f'{y_col} vs {x_col} 散点图'
                fig = px.scatter(df, x=x_col, y=y_col, **fig_kwargs)
                
            elif chart_type == 'line' and x_col and y_col:
                fig_kwargs['title'] = f'{y_col} 随 {x_col} 变化趋势'
                fig = px.line(df, x=x_col, y=y_col, **fig_kwargs)
                
            elif chart_type == 'bar' and x_col and y_col:
                # 柱状图默认按x轴分组求和
                agg_data = df.groupby(x_col)[y_col].sum().reset_index()
                fig_kwargs['title'] = f'{x_col} 分组 {y_col} 总和'
                fig = px.bar(agg_data, x=x_col, y=y_col, **fig_kwargs)
                
            elif chart_type == 'histogram' and x_col:
                fig_kwargs['title'] = f'{x_col} 分布直方图'
                fig = px.histogram(df, x=x_col, **fig_kwargs)
                
            elif chart_type == 'box' and x_col and y_col:
                fig_kwargs['title'] = f'{x_col} 分组 {y_col} 箱线图'
                fig = px.box(df, x=x_col, y=y_col, **fig_kwargs)
                
            elif chart_type == 'pie' and x_col:
                # 饼图统计x列的类别分布
                pie_data = df[x_col].value_counts().reset_index()
                pie_data.columns = [x_col, 'count']
                fig_kwargs['title'] = f'{x_col} 类别分布饼图'
                fig = px.pie(pie_data, values='count', names=x_col, **fig_kwargs)
            
            if 'fig' in locals():
                plot_html = pio.to_html(fig, full_html=False)
                
        except Exception as e:
            error = f'生成图表失败: {str(e)}'
            import traceback
            traceback.print_exc()
    
    return render_template('visualization.html',
                          filename=session['filename'],
                          numeric_columns=numeric_columns,
                          categorical_columns=categorical_columns,
                          plot_html=plot_html,
                          error=error)
# 分析功能页面
# 分析功能页面
@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    df = get_data()
    if df is None:
        return redirect(url_for('index'))
    
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    result_html = None
    error = None
    
    if request.method == 'POST':
        analysis_type = request.form.get('analysis_type')
        
        if analysis_type == 'kmeans':
            try:
                n_clusters = int(request.form.get('n_clusters', 3))
                selected_columns = request.form.getlist('columns')
                
                if len(selected_columns) < 2:
                    error = '请至少选择2个特征列进行聚类分析'
                else:
                    # 准备数据
                    X = df[selected_columns].dropna()
                    if len(X) < n_clusters:
                        error = f'数据样本数({len(X)})少于聚类数量({n_clusters})，请减少聚类数量或选择更多数据'
                    else:
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        # 执行K-Means聚类
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        clusters = kmeans.fit_predict(X_scaled)
                        
                        # 将聚类结果添加到数据中
                        X['Cluster'] = clusters.astype(str)
                        
                        # 生成可视化图表
                        if len(selected_columns) == 2:
                            fig = px.scatter(X, x=selected_columns[0], y=selected_columns[1],
                                           color='Cluster', title=f'K-Means聚类结果 (k={n_clusters})',
                                           template='plotly_white')
                        else:
                            # 多维数据使用前两个主成分可视化
                            from sklearn.decomposition import PCA
                            pca = PCA(n_components=2)
                            X_pca = pca.fit_transform(X_scaled)
                            pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
                            pca_df['Cluster'] = clusters.astype(str)
                            fig = px.scatter(pca_df, x='PC1', y='PC2',
                                           color='Cluster', title=f'K-Means聚类结果 (PCA降维, k={n_clusters})',
                                           template='plotly_white')
                        
                        # 计算每个聚类的中心
                        centers = scaler.inverse_transform(kmeans.cluster_centers_)
                        centers_df = pd.DataFrame(centers, columns=selected_columns)
                        centers_df['Cluster'] = range(n_clusters)
                        
                        result_html = f"""
                        <h3>K-Means聚类分析结果</h3>
                        <p>聚类数量: {n_clusters}</p>
                        <p>使用特征: {', '.join(selected_columns)}</p>
                        <p>有效样本数: {len(X)}</p>
                        <h4>聚类中心</h4>
                        {centers_df.to_html(classes='table table-striped', index=False)}
                        <h4>聚类可视化</h4>
                        {pio.to_html(fig, full_html=False)}
                        """
            except Exception as e:
                error = f'K-Means聚类分析失败: {str(e)}'
                import traceback
                traceback.print_exc()
        
        elif analysis_type == 'linear_regression':
            try:
                x_col = request.form.get('x_column')
                y_col = request.form.get('y_column')
                
                if not x_col or not y_col:
                    error = '请同时选择自变量和因变量'
                elif x_col == y_col:
                    error = '自变量和因变量不能相同'
                else:
                    # 准备数据
                    data = df[[x_col, y_col]].dropna()
                    if len(data) < 2:
                        error = '有效数据样本数不足，无法进行线性回归分析'
                    else:
                        X = data[[x_col]].values
                        y = data[y_col].values
                        
                        # 执行线性回归
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        # 预测
                        y_pred = model.predict(X)
                        
                        # 生成可视化图表
                        fig = px.scatter(data, x=x_col, y=y_col,
                                       title=f'线性回归: {y_col} = {model.coef_[0]:.4f} * {x_col} + {model.intercept_:.4f}',
                                       template='plotly_white')
                        fig.add_traces(px.line(x=data[x_col], y=y_pred, color_discrete_sequence=['red']).data)
                        
                        result_html = f"""
                        <h3>线性回归分析结果</h3>
                        <p>自变量: {x_col}</p>
                        <p>因变量: {y_col}</p>
                        <p>有效样本数: {len(data)}</p>
                        <p>回归方程: y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}</p>
                        <p>R² 决定系数: {model.score(X, y):.4f}</p>
                        <h4>回归可视化</h4>
                        {pio.to_html(fig, full_html=False)}
                        """
            except Exception as e:
                error = f'线性回归分析失败: {str(e)}'
                import traceback
                traceback.print_exc()
    
    return render_template('analysis.html',
                          filename=session['filename'],
                          numeric_columns=numeric_columns,
                          result_html=result_html,
                          error=error)
# 导出数据
@app.route('/export')
def export_data():
    df = get_data()
    if df is None:
        return redirect(url_for('index'))
    
    # 创建内存中的Excel文件
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Processed Data')
    
    output.seek(0)
    
    # 导出时使用原始文件名
    export_filename = f'processed_{session["filename"]}'
    # 确保导出文件名安全
    export_filename = safe_filename(export_filename)
    
    return send_file(output,
                     download_name=export_filename,
                     as_attachment=True)

# 重置会话
@app.route('/reset')
def reset():
    # 删除上传的文件
    if 'file_path' in session and os.path.exists(session['file_path']):
        try:
            os.remove(session['file_path'])
            print(f"已删除文件: {session['file_path']}")
        except Exception as e:
            print(f"删除文件失败: {str(e)}")
    
    # 清除session
    session.clear()
    
    return redirect(url_for('index'))

# 文件过大错误处理
@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template('index.html', error='文件太大了！请上传小于10MB的文件。'), 413

if __name__ == '__main__':
    app.run(debug=True, threaded=False)  # Windows下关闭多线程避免问题