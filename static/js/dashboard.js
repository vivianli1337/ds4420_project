document.addEventListener('DOMContentLoaded', function() {
    try {
        // Parse the chart data from the template
        const salesChartData = JSON.parse(document.getElementById('salesChart').getAttribute('data-chart'));
        const topItemsChartData = JSON.parse(document.getElementById('topItemsChart').getAttribute('data-chart'));
        const similarityMatrixData = JSON.parse(document.getElementById('similarityMatrix').getAttribute('data-matrix'));
        const itemSimilarityMatrixData = JSON.parse(document.getElementById('itemSimilarityMatrix').getAttribute('data-matrix'));
        const paymentChartData = JSON.parse(document.getElementById('paymentChart').getAttribute('data-chart'));
        const linePlotData = JSON.parse(document.getElementById('linePlot').getAttribute('data-plot'));
        const itemSalesChartData = JSON.parse(document.getElementById('itemSalesChart').getAttribute('data-chart'));
        const timeSeriesPlotData = JSON.parse(document.getElementById('timeSeriesPlot').getAttribute('data-plot'));
        const tsPlotData = JSON.parse(document.getElementById('tsPlot').getAttribute('data-plot'));
        
        // Render the charts
        Plotly.newPlot('salesChart', salesChartData.data, salesChartData.layout);
        Plotly.newPlot('topItemsChart', topItemsChartData.data, topItemsChartData.layout);
        Plotly.newPlot('paymentChart', paymentChartData.data, paymentChartData.layout);
        Plotly.newPlot('linePlot', linePlotData.data, linePlotData.layout);
        Plotly.newPlot('itemSalesChart', itemSalesChartData.data, itemSalesChartData.layout);
        Plotly.newPlot('timeSeriesPlot', timeSeriesPlotData.data, timeSeriesPlotData.layout);
        Plotly.newPlot('tsPlot', tsPlotData.data, tsPlotData.layout);
        
        // Create heatmap for customer similarity matrix
        const similarityData = [{
            z: similarityMatrixData,
            type: 'heatmap',
            colorscale: 'Viridis'
        }];
        
        const similarityLayout = {
            title: 'Customer Similarity Matrix',
            xaxis: {title: 'Customer ID'},
            yaxis: {title: 'Customer ID'}
        };
        
        Plotly.newPlot('similarityMatrix', similarityData, similarityLayout);
        
        // Create heatmap for item similarity matrix
        const itemSimilarityData = [{
            z: itemSimilarityMatrixData,
            type: 'heatmap',
            colorscale: 'Viridis'
        }];
        
        const itemSimilarityLayout = {
            title: 'Item Similarity Matrix',
            xaxis: {title: 'Item'},
            yaxis: {title: 'Item'}
        };
        
        Plotly.newPlot('itemSimilarityMatrix', itemSimilarityData, itemSimilarityLayout);
        
        // Add responsive behavior
        window.addEventListener('resize', function() {
            Plotly.Plots.resize('salesChart');
            Plotly.Plots.resize('topItemsChart');
            Plotly.Plots.resize('similarityMatrix');
            Plotly.Plots.resize('itemSimilarityMatrix');
            Plotly.Plots.resize('paymentChart');
            Plotly.Plots.resize('linePlot');
            Plotly.Plots.resize('itemSalesChart');
            Plotly.Plots.resize('timeSeriesPlot');
            Plotly.Plots.resize('tsPlot');
        });
    } catch (error) {
        console.error('Error rendering charts:', error);
        // Display error message to user
        document.querySelectorAll('.card-body').forEach(card => {
            if (!card.querySelector('div[id]').innerHTML) {
                card.innerHTML += '<div class="alert alert-danger">Error loading chart data</div>';
            }
        });
    }
}); 