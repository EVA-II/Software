import QtQuick
import QtQuick.Controls
import QtCharts

Item {
    id: root
    implicitHeight: 520
    property var chartPoints: []
    property string metricName: ""
    property string unitLabel: ""

    function saveSnapshot(targetPath) {
        if (!targetPath || targetPath === "")
            return
        chartView.grabToImage(function(result) {
            result.saveToFile(targetPath)
        })
    }

    function refreshData() {
        meanSeries.clear()
        upperSeries.clear()
        lowerSeries.clear()
        truthSeries.clear()

        if (!chartPoints || chartPoints.length === 0) {
            xAxis.min = 0
            xAxis.max = 1
            yAxis.min = 0
            yAxis.max = 1
            return
        }

        var minX = chartPoints[0].x
        var maxX = chartPoints[0].x
        // 初始化极值为极大/极小，避免第一组数据就小于0导致的基准错误
        var minY = Infinity
        var maxY = -Infinity

        // 极速追加点
        for (var i = 0; i < chartPoints.length; ++i) {
            var point = chartPoints[i]
            
            // 提取原始数据
            var pMean = point.mean
            var pUpper = point.upper
            var pLower = point.lower
            var pTruth = point.truth

            // 🌟 核心修改 1：针对特定指标进行物理边界截断
            // 注意：这里同时兼容了英文名和可能被翻译后的中文名，请根据你的实际显示调整
            if (metricName === "Wheel_Unloading_Rate" || metricName === "轮重减载率") {
                pLower = Math.max(0, pLower)
                pMean = Math.max(0, pMean)
                pUpper = Math.max(0, pUpper)
                if (pTruth !== undefined) pTruth = Math.max(0, pTruth)
            }
            
            meanSeries.append(point.x, pMean)
            upperSeries.append(point.x, pUpper)
            lowerSeries.append(point.x, pLower)
            
            if (pTruth !== undefined) {
                truthSeries.append(point.x, pTruth)
            }
            
            if (point.x < minX) minX = point.x
            if (point.x > maxX) maxX = point.x
            
            var currentTruth = (pTruth !== undefined) ? pTruth : pMean
            if (pLower < minY) minY = pLower
            if (pUpper > maxY) maxY = pUpper
            if (currentTruth < minY) minY = currentTruth
            if (currentTruth > maxY) maxY = currentTruth
        }

        var span = Math.max(maxY - minY, 1e-6)
        xAxis.min = minX
        xAxis.max = maxX
        
        // 🌟 核心修改 2：坐标轴下界也做对应的截断保护
        var calculatedMinY = minY - span * 0.08
        if (metricName === "Wheel_Unloading_Rate" || metricName === "轮重减载率") {
            yAxis.min = Math.max(0, calculatedMinY)
        } else {
            yAxis.min = calculatedMinY
        }
        
        yAxis.max = maxY + span * 0.08
    }

    onChartPointsChanged: refreshData()
    onMetricNameChanged: Qt.callLater(refreshData) // 👈 新增这行，名字变了也要重绘
    Component.onCompleted: refreshData()

    Rectangle {
        anchors.fill: parent
        radius: 14
        color: "#ffffff"
        border.color: "#d7dce5"

        ChartView {
            id: chartView
            anchors.fill: parent
            anchors.margins: 12
            antialiasing: true
            backgroundColor: "transparent"
            legend.visible: chartPoints && chartPoints.length > 0
            legend.alignment: Qt.AlignBottom
            title: metricName === "" ? "Prediction Chart" : metricName + (unitLabel === "" ? "" : " (" + unitLabel + ")")
            titleFont.pixelSize: 16

            ValueAxis {
                id: xAxis
                titleText: "里程/m" // 修改横坐标名称
                tickCount: 6
                labelsColor: "#233142"
                titleFont.pixelSize: 14 // 可选：稍微增大字体使其更清晰
            }

            ValueAxis {
                id: yAxis
                // 动态拼接：指标名称 + (单位)
                titleText: metricName === "" ? "值" : (metricName + (unitLabel === "" ? "" : " (" + unitLabel + ")"))
                tickCount: 6
                labelsColor: "#233142"
                titleFont.pixelSize: 14 // 可选：稍微增大字体
            }

            AreaSeries {
                id: bandSeries
                name: "95% interval"
                axisX: xAxis
                axisY: yAxis
                color: "#dceef8"
                borderColor: "#8fb8cf"
                opacity: 0.75
                upperSeries: LineSeries { id: upperSeries; color: "#8fb8cf" }
                lowerSeries: LineSeries { id: lowerSeries; color: "#8fb8cf" }
            }

            LineSeries {
                id: meanSeries
                name: "Predicted mean"
                axisX: xAxis
                axisY: yAxis
                color: "#0f6a9b"
                width: 2.2
            }

            LineSeries {
                id: truthSeries
                name: "Ground truth"
                axisX: xAxis
                axisY: yAxis
                color: "#d1495b"
                width: 1.8
            }
        }

        Column {
            anchors.centerIn: parent
            spacing: 10
            visible: !chartPoints || chartPoints.length === 0

            Label {
                anchors.horizontalCenter: parent.horizontalCenter
                text: "No prediction curve yet"
                font.pixelSize: 22
                font.bold: true
                color: "#20333f"
            }

            Label {
                anchors.horizontalCenter: parent.horizontalCenter
                text: "Run inference to populate the chart and enable metric export."
                color: "#55646e"
            }
        }
    }
}
