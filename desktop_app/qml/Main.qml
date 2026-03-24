import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: window
    width: 1500
    height: 960
    visible: true
    title: "Bridge Assessment Workbench"
    color: "#eef1f4"

    readonly property var vm: (typeof workbench !== "undefined" && workbench !== null) ? workbench : ({
        result: {},
        lastTrainingRun: {},
        assets: [],
        recentTasks: [],
        task: {},
        health: {},
        appSettings: {},
        logs: "",
        lastError: "",
        busy: false,
        selectedMetric: "",
        metricNames: [],
        chartPoints: []
    })

    readonly property bool hasPrediction: vm.result && vm.result.series ? vm.result.series.length > 0 : false
    readonly property bool hasTrainingRun: vm.lastTrainingRun && vm.lastTrainingRun.version ? true : false
    readonly property bool hasAssets: vm.assets ? vm.assets.length > 0 : false
    readonly property bool hasTaskHistory: vm.recentTasks ? vm.recentTasks.length > 0 : false

    function currentSeries() {
        var seriesList = vm.result && vm.result.series ? vm.result.series : []
        for (var i = 0; i < seriesList.length; ++i) {
            if (seriesList[i].name === vm.selectedMetric)
                return seriesList[i]
        }
        return seriesList.length > 0 ? seriesList[0] : null
    }

    function maxValue(values) {
        if (!values || values.length === 0)
            return "-"
        var max = values[0]
        for (var i = 1; i < values.length; ++i)
            max = Math.max(max, values[i])
        return Number(max).toFixed(4)
    }

    background: Rectangle {
        gradient: Gradient {
            GradientStop { position: 0.0; color: "#f5f1e8" }
            GradientStop { position: 0.35; color: "#eef1f4" }
            GradientStop { position: 1.0; color: "#dde5ea" }
        }
    }

    header: Rectangle {
        height: 72
        color: "#20333f"
        border.color: "#425a68"

        RowLayout {
            anchors.fill: parent
            anchors.margins: 18
            spacing: 16

            ColumnLayout {
                spacing: 2
                Label {
                    text: "Bridge Assessment Workbench"
                    color: "#f5f1e8"
                    font.pixelSize: 24
                    font.family: "Segoe UI"
                    font.bold: true
                }
                Label {
                    text: "PySide6 + QML + Python core runtime"
                    color: "#b7c7d1"
                    font.pixelSize: 12
                }
            }

            Item { Layout.fillWidth: true }

            Rectangle {
                radius: 18
                color: vm.health.asset_ready ? "#d6f2e2" : "#f5d8d8"
                Layout.preferredHeight: 38
                Layout.preferredWidth: 400

                Label {
                    anchors.centerIn: parent
                    text: (vm.health.asset_ready ? "Assets ready" : "Assets unavailable") + " | Version: " + (vm.health.model_version || "-") + " | Device: " + (vm.health.device || "-")
                    color: "#20333f"
                    font.pixelSize: 12
                    font.bold: true
                }
            }
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 18
        spacing: 14

        Rectangle {
            Layout.fillWidth: true
            implicitHeight: taskCardContent.implicitHeight + 32
            radius: 14
            color: "#ffffff"
            border.color: "#d1d8df"

            ColumnLayout {
                id: taskCardContent
                anchors.fill: parent
                anchors.margins: 16
                spacing: 10

                ProgressBar {
                    Layout.fillWidth: true
                    from: 0
                    to: 100
                    value: vm.task.progress || 0
                    visible: vm.busy || ((vm.task.status || "") !== "")
                }

                RowLayout {
                    Layout.fillWidth: true
                    Label {
                        text: "Task: " + (vm.task.kind || "idle") + " | Status: " + (vm.task.status || "idle") + " | Progress: " + (vm.task.progress || 0) + "%"
                        color: "#20333f"
                        font.pixelSize: 13
                        font.bold: true
                    }
                    Item { Layout.fillWidth: true }
                    BusyIndicator {
                        running: vm.busy
                        visible: running
                    }
                }

                Label {
                    Layout.fillWidth: true
                    text: vm.task.message || "Ready"
                    color: "#55646e"
                    elide: Text.ElideRight
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            visible: vm.lastError !== ""
            implicitHeight: errorLabel.implicitHeight + 28
            color: "#fff1f1"
            border.color: "#d67878"
            radius: 12

            Label {
                id: errorLabel
                anchors.fill: parent
                anchors.margins: 14
                wrapMode: Text.Wrap
                text: vm.lastError
                color: "#7a1f1f"
            }
        }

        TabBar {
            id: workspaces
            Layout.fillWidth: true
            spacing: 8
            background: Rectangle { color: "transparent" }
            TabButton { text: "Inference" }
            TabButton { text: "Analysis" }
            TabButton { text: "Training" }
            TabButton { text: "Assets" }
            TabButton { text: "Settings && Logs" }
        }

        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: workspaces.currentIndex

            ScrollView {
                id: inferenceScroll
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                contentWidth: availableWidth

                Column {
                    width: inferenceScroll.availableWidth
                    spacing: 16

                    RowLayout {
                        width: parent.width
                        spacing: 16

                        Rectangle {
                            Layout.preferredWidth: 440
                            Layout.alignment: Qt.AlignTop
                            implicitHeight: inferenceForm.implicitHeight + 36
                            radius: 16
                            color: "#ffffff"
                            border.color: "#d1d8df"

                            ColumnLayout {
                                id: inferenceForm
                                anchors.fill: parent
                                anchors.margins: 18
                                spacing: 12

                                Label {
                                    text: "Inference Workbench"
                                    font.pixelSize: 22
                                    font.bold: true
                                    color: "#20333f"
                                }

                                Label {
                                    Layout.fillWidth: true
                                    text: "Upload an Excel, CSV, or JSON dataset, then run direct probabilistic inference with the active model assets."
                                    wrapMode: Text.Wrap
                                    color: "#55646e"
                                }

                                TextField {
                                    id: inferencePathField
                                    Layout.fillWidth: true
                                    placeholderText: "Dataset file path"
                                }

                                Button {
                                    text: "Choose Dataset"
                                    Layout.preferredWidth: 150
                                    onClicked: {
                                        var selected = vm.browseInferenceFile()
                                        if (selected)
                                            inferencePathField.text = selected
                                    }
                                }

                                TextField {
                                    id: scenarioIdField
                                    Layout.fillWidth: true
                                    placeholderText: "scenario_id for multi-scenario workbooks"
                                }

                                ComboBox {
                                    id: speedLevelBox
                                    Layout.fillWidth: true
                                    model: ["", "slow", "medium", "high"]
                                }

                                TextField {
                                    id: trainFeaturesField
                                    Layout.fillWidth: true
                                    placeholderText: "Optional train features, e.g. 120,1.2,1.0"
                                }

                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 10

                                    Button {
                                        text: "Run Inference"
                                        Layout.fillWidth: true
                                        enabled: !vm.busy
                                        onClicked: vm.runInference(inferencePathField.text, scenarioIdField.text, speedLevelBox.currentText, trainFeaturesField.text)
                                    }

                                    Button {
                                        text: "Cancel"
                                        Layout.preferredWidth: 110
                                        enabled: vm.busy
                                        onClicked: vm.cancelActiveTask()
                                    }
                                }

                                Flow {
                                    Layout.fillWidth: true
                                    spacing: 10

                                    Button {
                                        text: "Export JSON"
                                        onClicked: {
                                            var selected = vm.browseExportJsonPath()
                                            if (selected)
                                                vm.exportLastResultJson(selected)
                                        }
                                    }

                                    Button {
                                        text: "Export CSV"
                                        onClicked: {
                                            var selected = vm.browseExportCsvPath()
                                            if (selected)
                                                vm.exportLastResultCsv(selected)
                                        }
                                    }

                                    Button {
                                        text: "Snapshot PNG"
                                        onClicked: {
                                            var selected = vm.browseExportImagePath()
                                            if (selected)
                                                previewChart.saveSnapshot(selected)
                                        }
                                    }
                                }

                                Rectangle {
                                    Layout.fillWidth: true
                                    implicitHeight: summaryGrid.implicitHeight + 28
                                    color: "#f5f7f9"
                                    radius: 12
                                    border.color: "#d7dce5"

                                    GridLayout {
                                        id: summaryGrid
                                        anchors.fill: parent
                                        anchors.margins: 14
                                        columns: 2
                                        rowSpacing: 8
                                        columnSpacing: 10

                                        Label { text: "Assessment"; color: "#55646e" }
                                        Label { text: vm.result.status ? vm.result.status.label : "-"; font.bold: true }
                                        Label { text: "Scenario"; color: "#55646e" }
                                        Label { text: vm.result.meta ? vm.result.meta.scenario_id : "-" }
                                        Label { text: "Node count"; color: "#55646e" }
                                        Label { text: vm.result.meta ? vm.result.meta.node_count : "-" }
                                        Label { text: "Model count"; color: "#55646e" }
                                        Label { text: vm.result.meta ? vm.result.meta.model_count : "-" }
                                        Label { text: "Reason"; color: "#55646e" }
                                        Label {
                                            text: vm.result.status ? vm.result.status.reason : "-"
                                            wrapMode: Text.Wrap
                                            Layout.fillWidth: true
                                        }
                                    }
                                }
                            }
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            Layout.alignment: Qt.AlignTop
                            implicitHeight: chartCard.implicitHeight + 36
                            radius: 16
                            color: "#ffffff"
                            border.color: "#d1d8df"

                            ColumnLayout {
                                id: chartCard
                                anchors.fill: parent
                                anchors.margins: 18
                                spacing: 12

                                RowLayout {
                                    Layout.fillWidth: true
                                    Label {
                                        text: "Prediction Preview"
                                        font.pixelSize: 22
                                        font.bold: true
                                        color: "#20333f"
                                    }
                                    Item { Layout.fillWidth: true }
                                    ComboBox {
                                        id: metricSelector
                                        Layout.preferredWidth: 240
                                        // 过滤掉不想展示的指标
                                        model: vm.metricNames ? vm.metricNames.filter(function(name) { return name !== "Suspension_Force"; }) : []
                                        enabled: model.length > 0
                                        onActivated: vm.setSelectedMetric(currentText)
                                    }
                                }

                                PredictionChartPane {
                                    id: previewChart
                                    Layout.fillWidth: true
                                    Layout.preferredHeight: 520
                                    chartPoints: vm.chartPoints
                                    metricName: vm.selectedMetric
                                    unitLabel: window.currentSeries() ? window.currentSeries().unit : ""
                                }
                            }
                        }
                    }
                }
            }
            ScrollView {
                id: analysisScroll
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                contentWidth: availableWidth

                Column {
                    width: analysisScroll.availableWidth
                    spacing: 16

                    Rectangle {
                        width: parent.width
                        implicitHeight: analysisCard.implicitHeight + 36
                        radius: 16
                        color: "#ffffff"
                        border.color: "#d1d8df"

                        ColumnLayout {
                            id: analysisCard
                            anchors.fill: parent
                            anchors.margins: 18
                            spacing: 12

                            Label {
                                text: "Result Analysis"
                                font.pixelSize: 22
                                font.bold: true
                                color: "#20333f"
                            }

                            Label {
                                Layout.fillWidth: true
                                text: hasPrediction ? "Metric summaries are generated from the latest inference result." : "Run inference first. This page will then summarize each output metric, uncertainty envelope, and available ground-truth coverage."
                                wrapMode: Text.Wrap
                                color: "#55646e"
                            }

                            Rectangle {
                                visible: !hasPrediction
                                Layout.fillWidth: true
                                implicitHeight: 120
                                radius: 12
                                color: "#f5f7f9"
                                border.color: "#d7dce5"

                                Label {
                                    anchors.centerIn: parent
                                    text: "No prediction result yet"
                                    color: "#55646e"
                                    font.pixelSize: 18
                                }
                            }

                            Column {
                                id: analysisList
                                Layout.fillWidth: true
                                spacing: 12
                                visible: hasPrediction

                                Repeater {
                                    model: hasPrediction ? vm.result.series.filter(function(s) { return s.name !== "Suspension_Force"; }) : []
                                    delegate: Rectangle {
                                        width: analysisList.width
                                        implicitHeight: metricCardLayout.implicitHeight + 28
                                        radius: 12
                                        color: "#f5f7f9"
                                        border.color: "#d7dce5"

                                        GridLayout {
                                            id: metricCardLayout
                                            anchors.fill: parent
                                            anchors.margins: 14
                                            columns: 4
                                            columnSpacing: 12
                                            rowSpacing: 8

                                            Label { text: modelData.name; font.bold: true; color: "#20333f" }
                                            Label { text: "Peak mean: " + window.maxValue(modelData.mean) }
                                            Label { text: "Peak upper: " + window.maxValue(modelData.upper_95) }
                                            Label { text: "Peak total var: " + window.maxValue(modelData.total_var) }
                                            Label { text: "Unit: " + modelData.unit; color: "#55646e" }
                                            Label { text: "Sampled points: " + (modelData.points ? modelData.points.length : 0) }
                                            Label { text: "Ground truth: " + (modelData.ground_truth ? "yes" : "no") }
                                            Label { text: "Ready"; color: "#0f6a9b" }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            ScrollView {
                id: trainingScroll
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                contentWidth: availableWidth

                Column {
                    width: trainingScroll.availableWidth
                    spacing: 16

                    Rectangle {
                        width: parent.width
                        implicitHeight: trainingForm.implicitHeight + 36
                        radius: 16
                        color: "#ffffff"
                        border.color: "#d1d8df"

                        ColumnLayout {
                            id: trainingForm
                            anchors.fill: parent
                            anchors.margins: 18
                            spacing: 12

                            Label {
                                text: "Training Workbench"
                                font.pixelSize: 22
                                font.bold: true
                                color: "#20333f"
                            }

                            Label {
                                Layout.fillWidth: true
                                text: "This page is wired to the new desktop training service. Select a dataset and output folder, then launch a local training run in the background."
                                wrapMode: Text.Wrap
                                color: "#55646e"
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                TextField {
                                    id: trainingDatasetField
                                    Layout.fillWidth: true
                                    placeholderText: "Training dataset path"
                                }
                                Button {
                                    text: "Choose Dataset"
                                    Layout.preferredWidth: 150
                                    onClicked: {
                                        var selected = vm.browseTrainingDataset()
                                        if (selected)
                                            trainingDatasetField.text = selected
                                    }
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                TextField {
                                    id: trainingOutputField
                                    Layout.fillWidth: true
                                    placeholderText: "Output directory"
                                    text: vm.appSettings.last_output_dir || ""
                                }
                                Button {
                                    text: "Choose Folder"
                                    Layout.preferredWidth: 150
                                    onClicked: {
                                        var selected = vm.browseTrainingOutputDir()
                                        if (selected)
                                            trainingOutputField.text = selected
                                    }
                                }
                            }

                            GridLayout {
                                Layout.fillWidth: true
                                columns: 2
                                rowSpacing: 10
                                columnSpacing: 14

                                Label { text: "Device"; color: "#55646e" }
                                ComboBox { id: deviceBox; Layout.fillWidth: true; model: ["cpu", "cuda"] }

                                Label { text: "Train size"; color: "#55646e" }
                                TextField { id: trainSizeField; Layout.fillWidth: true; text: "0.70"; placeholderText: "0-1" }

                                Label { text: "Batch size"; color: "#55646e" }
                                SpinBox { id: batchSizeBox; from: 1; to: 256; value: 16; Layout.fillWidth: true }

                                Label { text: "Ensemble models"; color: "#55646e" }
                                SpinBox { id: modelCountBox; from: 1; to: 10; value: 5; Layout.fillWidth: true }

                                Label { text: "Epochs"; color: "#55646e" }
                                SpinBox { id: epochBox; from: 10; to: 5000; value: 300; Layout.fillWidth: true }

                                Label { text: "Patience"; color: "#55646e" }
                                SpinBox { id: patienceBox; from: 1; to: 500; value: 80; Layout.fillWidth: true }

                                Label { text: "Warmup epochs"; color: "#55646e" }
                                SpinBox { id: warmupBox; from: 0; to: 1000; value: 40; Layout.fillWidth: true }
                            }

                            Button {
                                text: "Start Training"
                                Layout.preferredWidth: 180
                                enabled: !vm.busy
                                onClicked: vm.startTraining(
                                    trainingDatasetField.text,
                                    trainingOutputField.text,
                                    deviceBox.currentText,
                                    parseFloat(trainSizeField.text),
                                    batchSizeBox.value,
                                    modelCountBox.value,
                                    epochBox.value,
                                    patienceBox.value,
                                    warmupBox.value
                                )
                            }
                        }
                    }
                    Rectangle {
                        width: parent.width
                        implicitHeight: trainingSummary.implicitHeight + 36
                        radius: 16
                        color: "#ffffff"
                        border.color: "#d1d8df"

                        ColumnLayout {
                            id: trainingSummary
                            anchors.fill: parent
                            anchors.margins: 18
                            spacing: 10

                            Label {
                                text: "Last Training Run"
                                font.pixelSize: 20
                                font.bold: true
                                color: "#20333f"
                            }

                            Label {
                                Layout.fillWidth: true
                                visible: !hasTrainingRun
                                text: "No training run has completed yet. After the first successful run, this panel will show the registered version, output folder, and final status."
                                wrapMode: Text.Wrap
                                color: "#55646e"
                            }

                            ColumnLayout {
                                Layout.fillWidth: true
                                visible: hasTrainingRun
                                spacing: 8
                                Label { text: "Version: " + (vm.lastTrainingRun.version || "-") }
                                Label { text: "Output: " + (vm.lastTrainingRun.output_dir || "-"); wrapMode: Text.Wrap }
                                Label { text: "Status: " + (vm.lastTrainingRun.status || "-") }
                                Label { text: "Message: " + (vm.lastTrainingRun.message || "-"); wrapMode: Text.Wrap }
                            }
                        }
                    }
                }
            }

            ScrollView {
                id: assetsScroll
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                contentWidth: availableWidth

                Column {
                    width: assetsScroll.availableWidth
                    spacing: 16

                    Rectangle {
                        width: parent.width
                        implicitHeight: assetsCard.implicitHeight + 36
                        radius: 16
                        color: "#ffffff"
                        border.color: "#d1d8df"

                        ColumnLayout {
                            id: assetsCard
                            anchors.fill: parent
                            anchors.margins: 18
                            spacing: 12

                            RowLayout {
                                Layout.fillWidth: true
                                Label {
                                    text: "Asset Library"
                                    font.pixelSize: 22
                                    font.bold: true
                                    color: "#20333f"
                                }
                                Item { Layout.fillWidth: true }
                                Button { text: "Refresh"; onClicked: vm.refreshAssets() }
                            }

                            Label {
                                Layout.fillWidth: true
                                text: hasAssets ? "Installed model packages are listed below. You can switch the active version at runtime." : "No assets are indexed yet."
                                wrapMode: Text.Wrap
                                color: "#55646e"
                            }

                            Rectangle {
                                visible: !hasAssets
                                Layout.fillWidth: true
                                implicitHeight: 120
                                radius: 12
                                color: "#f5f7f9"
                                border.color: "#d7dce5"

                                Label {
                                    anchors.centerIn: parent
                                    text: "No asset versions available"
                                    color: "#55646e"
                                    font.pixelSize: 18
                                }
                            }

                            Column {
                                id: assetList
                                Layout.fillWidth: true
                                spacing: 12
                                visible: hasAssets

                                Repeater {
                                    model: hasAssets ? vm.assets : []
                                    delegate: Rectangle {
                                        width: assetList.width
                                        implicitHeight: assetContent.implicitHeight + 28
                                        radius: 12
                                        color: modelData.is_active ? "#e8f4ea" : "#f5f7f9"
                                        border.color: modelData.is_active ? "#6aa67b" : "#d7dce5"

                                        ColumnLayout {
                                            id: assetContent
                                            anchors.fill: parent
                                            anchors.margins: 14
                                            spacing: 8

                                            RowLayout {
                                                Layout.fillWidth: true
                                                Label {
                                                    text: modelData.version
                                                    font.bold: true
                                                    color: "#20333f"
                                                }
                                                Item { Layout.fillWidth: true }
                                                Label {
                                                    text: modelData.is_active ? "Active" : "Inactive"
                                                    color: modelData.is_active ? "#2c7a4b" : "#55646e"
                                                }
                                                Button {
                                                    text: modelData.is_active ? "Current" : "Activate"
                                                    enabled: !modelData.is_active
                                                    onClicked: vm.activateAsset(modelData.version)
                                                }
                                            }

                                            Label { text: "Models: " + modelData.model_count; color: "#55646e" }
                                            Label { text: "Root: " + modelData.root_dir; wrapMode: Text.WrapAnywhere; Layout.fillWidth: true }
                                            Label { text: "Notes: " + (modelData.notes || "-"); wrapMode: Text.Wrap; Layout.fillWidth: true }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            ScrollView {
                id: settingsScroll
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                contentWidth: availableWidth

                Column {
                    width: settingsScroll.availableWidth
                    spacing: 16

                    Rectangle {
                        width: parent.width
                        implicitHeight: settingsCard.implicitHeight + 36
                        radius: 16
                        color: "#ffffff"
                        border.color: "#d1d8df"

                        ColumnLayout {
                            id: settingsCard
                            anchors.fill: parent
                            anchors.margins: 18
                            spacing: 12

                            Label {
                                text: "Settings"
                                font.pixelSize: 22
                                font.bold: true
                                color: "#20333f"
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                Label { text: "Chart sample limit"; color: "#55646e" }
                                SpinBox {
                                    id: sampleLimitBox
                                    from: 50
                                    to: 5000
                                    editable: true  // 👈 新增这一行，允许直接键盘输入
                                    value: vm.appSettings.sample_limit || 600
                                    onValueModified: vm.updateSampleLimit(value)
                                }
                            }
                        }
                    }

                    Rectangle {
                        width: parent.width
                        implicitHeight: historyCard.implicitHeight + 36
                        radius: 16
                        color: "#ffffff"
                        border.color: "#d1d8df"

                        ColumnLayout {
                            id: historyCard
                            anchors.fill: parent
                            anchors.margins: 18
                            spacing: 10

                            RowLayout {
                                Layout.fillWidth: true
                                Label {
                                    text: "Recent Tasks"
                                    font.pixelSize: 20
                                    font.bold: true
                                    color: "#20333f"
                                }
                                Item { Layout.fillWidth: true }
                                Button { text: "Refresh Overview"; onClicked: vm.refreshOverview() }
                            }

                            Label {
                                Layout.fillWidth: true
                                visible: !hasTaskHistory
                                text: "No task history yet. Once you run inference or training, task entries will be listed here."
                                wrapMode: Text.Wrap
                                color: "#55646e"
                            }

                            Column {
                                id: historyList
                                Layout.fillWidth: true
                                spacing: 10
                                visible: hasTaskHistory

                                Repeater {
                                    model: hasTaskHistory ? vm.recentTasks : []
                                    delegate: Rectangle {
                                        width: historyList.width
                                        implicitHeight: historyItemContent.implicitHeight + 24
                                        radius: 10
                                        color: "#f5f7f9"
                                        border.color: "#d7dce5"

                                        ColumnLayout {
                                            id: historyItemContent
                                            anchors.fill: parent
                                            anchors.margins: 12
                                            spacing: 4

                                            Label { text: modelData.kind + " | " + modelData.status + " | " + modelData.progress + "%"; font.bold: true }
                                            Label { text: modelData.message || "-"; wrapMode: Text.Wrap }
                                            Label { text: modelData.updated_at || "-"; color: "#55646e" }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    Rectangle {
                        width: parent.width
                        implicitHeight: 360
                        radius: 16
                        color: "#ffffff"
                        border.color: "#d1d8df"

                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 18
                            spacing: 10

                            RowLayout {
                                Layout.fillWidth: true
                                Label {
                                    text: "Logs"
                                    font.pixelSize: 20
                                    font.bold: true
                                    color: "#20333f"
                                }
                                Item { Layout.fillWidth: true }
                                Button { text: "Clear"; onClicked: vm.clearLogs() }
                            }

                            TextArea {
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                readOnly: true
                                text: vm.logs
                                wrapMode: Text.WrapAnywhere
                                font.family: "Consolas"
                                placeholderText: "Logs will appear here during inference and training."
                            }
                        }
                    }
                }
            }
        }
    }
}
