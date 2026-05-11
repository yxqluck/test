// 通用脚本
document.addEventListener('DOMContentLoaded', function() {
    // 自动调整图表大小
    window.addEventListener('resize', function() {
        const plots = document.querySelectorAll('.js-plotly-plot');
        plots.forEach(plot => {
            Plotly.Plots.resize(plot);
        });
    });
});