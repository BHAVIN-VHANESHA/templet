'use strict';
setTimeout(function() {
    (function () {
        var options = {
            chart: {
                height: 300,
                type: 'line',
                zoom: {
                    enabled: false
                }
            },
            dataLabels: {
                enabled: false,
                width: 2,
            },
            stroke: {
                curve: 'straight',
            },
            colors: ["#4099ff"],
            fill: {
                type: "gradient",
                gradient: {
                    shade: 'light'
                },
            },
            series: [{
                name: "Desktops",
                data: [10, 41, 35, 51, 49, 62, 69, 91, 148]
            }],
            title: {
                text: 'Product Trends by Month',
                align: 'left'
            },
            grid: {
                row: {
                    colors: ['#f3f6ff', 'transparent'], // takes an array which will be repeated on columns
                    opacity: 0.5
                },
            },
            xaxis: {
                categories: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'],
            }
        }
        var chart = new ApexCharts(
            document.querySelector("#line-chart-1"),
            options
        );
        chart.render();
    })();
    (function () {
        var options = {
            chart: {
                height: 350,
                type: 'area',
            },
            dataLabels: {
                enabled: false
            },
            stroke: {
                curve: 'smooth'
            },
            colors: ["#FFB64D", "#FF5370", "#707B7C", "#E74C3C", "#DC7633", "#5D6D7E", "#28B463", "#A569BD", "#0B5345",
            "#F4D03F", "#FA8072"],
            series: [{
                name: 'BLOCK JOINING MOTAR',
                data: [31, 40, 28, 51, 42, 109, 100]
            }, {
                name: 'BROMEL FIX TYPE-3',
                data: [11, 32, 45, 34, 52, 41]
            },  {
                name: 'BROMEL FIX TYPE–1',
                data: [11, 32, 32, 34, 52, 41]
            },  {
                name: 'BROMEL FIX TYPE–2',
                data: [11, 45, 32, 34, 52, 41]
            },  {
                name: 'EPOXY REGULAR(1 KG)',
                data: [32, 45, 32, 34, 52, 41]
            },  {
                name: 'EPOXY REGULAR(5 KG)',
                data: [11, 55, 45, 32, 34, 52, 41]
            },  {
                name: 'SPARKAL COPPER',
                data: [11, 32, 45, 32, 34, 12, 41]
            },  {
                name: 'SPARKAL GOLDEN',
                data: [11, 32, 45, 32, 40, 52, 41]
            },  {
                name: 'SPARKAL SILVER',
                data: [11, 32, 22, 32, 34, 52, 41]
            },  {
                name: 'WHITE SBR(1 LTR)',
                data: [11, 37, 45, 32, 34, 52, 41]
            },  {
                name: 'WHITE SBR(5 LTR)',
                data: [10, 32, 45, 32, 34, 52, 41]
            }],

            xaxis: {
                type: 'datetime',
                categories: [],
            },
            tooltip: {
                x: {
                    format: 'MM-yyyy'
                },
            }
        }

        var chart = new ApexCharts(
            document.querySelector("#area-chart-1"),
            options
        );

        chart.render();
    })();
    (function () {
        var options = {
            chart: {
                height: 350,
                type: 'bar',
            },
            plotOptions: {
                bar: {
                    horizontal: false,
                    columnWidth: '55%',
                    endingShape: 'rounded'
                },
            },
            dataLabels: {
                enabled: false
            },
            colors: ["#0e9e4a", "#4099ff", "#FF5370"],
            stroke: {
                show: true,
                width: 2,
                colors: ['transparent']
            },
            fill: {
                type: 'gradient',
                gradient: {
                    shade: 'light',
                    type: "vertical",
                    shadeIntensity: 0.25,
                    inverseColors: true,
                    opacityFrom: 1,
                    opacityTo: 0.7,
                    stops: [50, 100]
                },
            },
            series: [{
                name: 'Net Profit',
                data: [44, 55, 57, 56, 61, 58, 63]
            }, {
                name: 'Revenue',
                data: [76, 85, 101, 98, 87, 105, 91]
            }, {
                name: 'Free Cash Flow',
                data: [35, 41, 36, 26, 45, 48, 52]
            }],
            xaxis: {
                categories: ['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug'],
            },
            yaxis: {
                title: {
                    text: '$ (thousands)'
                }
            },
            tooltip: {
                y: {
                    formatter: function(val) {
                        return "$ " + val + " thousands"
                    }
                }
            }
        }
        var chart = new ApexCharts(
            document.querySelector("#bar-chart-1"),
            options
        );
        chart.render();
    })();
    (function () {
        var options = {
            chart: {
                height: 350,
                type: 'bar',
            },
            plotOptions: {
                bar: {
                    horizontal: true,
                    dataLabels: {
                        position: 'top',
                    },
                }
            },
            colors: ["#4099ff", "#0e9e4a"],
            dataLabels: {
                enabled: true,
                offsetX: -6,
                style: {
                    fontSize: '12px',
                    colors: ['#fff']
                }
            },
            stroke: {
                show: true,
                width: 1,
                colors: ['#fff']
            },
            fill: {
                type: 'gradient',
                gradient: {
                    shade: 'light',
                    type: "horizontal",
                    shadeIntensity: 0.25,
                    inverseColors: true,
                    opacityFrom: 0.8,
                    opacityTo: 1,
                    stops: [0, 100]
                },
            },
            series: [{
                data: [44, 55, 41, 64, 22, 43, 21]
            }, {
                data: [53, 32, 33, 52, 13, 44, 32]
            }],
            xaxis: {
                categories: [2001, 2002, 2003, 2004, 2005, 2006, 2007],
            },
        }
        var chart = new ApexCharts(
            document.querySelector("#bar-chart-3"),
            options
        );
        chart.render();
    })();
    (function () {
        var options = {
            chart: {
                height: 320,
                type: 'pie',
            },
            labels: ['Team A', 'Team B', 'Team C', 'Team D', 'Team E'],
            series: [44, 55, 13, 43, 22],
            colors: ["#4099ff", "#0e9e4a", "#00bcd4", "#FFB64D", "#FF5370"],
            legend: {
                show: true,
                position: 'bottom',
            },
            fill: {
                type: 'gradient',
                gradient: {
                    shade: 'light',
                    inverseColors: true,
                }
            },
            dataLabels: {
                enabled: true,
                dropShadow: {
                    enabled: false,
                }
            },
            responsive: [{
                breakpoint: 480,
                options: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }]
        }
        var chart = new ApexCharts(
            document.querySelector("#pie-chart-1"),
            options
        );
        chart.render();
    })();
    (function () {
        var options = {
            chart: {
                height: 320,
                type: 'donut',
            },
            series: [44, 55, 41, 17, 15],
            colors: ["#4099ff", "#0e9e4a", "#00bcd4", "#FFB64D", "#FF5370"],
            fill: {
                type: 'gradient',
                gradient: {
                    shade: 'light',
                    inverseColors: true,
                }
            },
            legend: {
                show: true,
                position: 'bottom',
            },
            plotOptions: {
                pie: {
                    donut: {
                        labels: {
                            show: true,
                            name: {
                                show: true
                            },
                            value: {
                                show: true
                            }
                        }
                    }
                }
            },
            dataLabels: {
                enabled: true,
                dropShadow: {
                    enabled: false,
                }
            },
            responsive: [{
                breakpoint: 480,
                options: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }]
        }
        var chart = new ApexCharts(
            document.querySelector("#pie-chart-2"),
            options
        );
        chart.render();
    })();
}, 700);