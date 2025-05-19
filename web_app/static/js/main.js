// Initialize Socket.IO connection
const socket = io();

// Initialize accuracy chart
let accuracyChart = null;
let trainAccData = [];
let testAccData = [];
let epochLabels = [];

function initializeAccuracyChart() {
    const ctx = document.getElementById('lossChart').getContext('2d');
    accuracyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: epochLabels,
            datasets: [
                {
                    label: '训练准确率',
                    data: trainAccData,
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                },
                {
                    label: '验证准确率',
                    data: testAccData,
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: '准确率 (%)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Epoch'
                    }
                }
            }
        }
    });
}

// Socket.IO event handlers
socket.on('training_progress', function(data) {
    updateTrainingProgress(data);
});

// Update training progress
function updateTrainingProgress(data) {
    const progressBar = document.querySelector('#trainingProgress .progress-bar');
    const currentEpoch = document.getElementById('currentEpoch');
    const currentAccuracy = document.getElementById('currentAccuracy');
    const currentLR = document.getElementById('currentLR');
    
    // Calculate progress percentage
    const progress = (data.epoch / data.total_epochs) * 100;
    
    // Update progress bar
    progressBar.style.width = `${progress}%`;
    progressBar.setAttribute('aria-valuenow', progress);
    
    // Update text information
    currentEpoch.textContent = `Epoch: ${data.epoch}/${data.total_epochs}`;
    currentAccuracy.textContent = `准确率: ${data.test_acc.toFixed(2)}%`;
    currentLR.textContent = data.lr.toFixed(6);

    // Update accuracy chart
    if (accuracyChart === null) {
        initializeAccuracyChart();
    }
    
    // Add new data points
    epochLabels.push(data.epoch);
    trainAccData.push(data.train_acc);
    testAccData.push(data.test_acc);
    
    // Update chart
    accuracyChart.data.labels = epochLabels;
    accuracyChart.data.datasets[0].data = trainAccData;
    accuracyChart.data.datasets[1].data = testAccData;
    accuracyChart.update();
}

// Show toast notification
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    const toastBody = toast.querySelector('.toast-body');
    
    toastBody.textContent = message;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// Handle dataset selection for training
document.getElementById('trainDataset').addEventListener('change', function() {
    const dataset = this.value;
    if (!dataset) return;
    
    const is3D = dataset.includes('3d');
    const modelSelect = document.getElementById('trainModel');
    
    // Clear existing options
    modelSelect.innerHTML = '<option value="">Select Model</option>';
    
    // Add new options based on dataset type
    const models = is3D ? ['cnn', 'resnet18', 'resnet34'] : ['cnn', 'resnet18', 'resnet34'];
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model;
        modelSelect.appendChild(option);
    });
    
    // Set default training parameters based on dataset type
    if (is3D) {
        document.getElementById('batchSize').value = '32';
        document.getElementById('epochs').value = '20';
        document.getElementById('learningRate').value = '0.0001';
    } else {
        document.getElementById('batchSize').value = '128';
        document.getElementById('epochs').value = '10';
        document.getElementById('learningRate').value = '0.001';
    }
});

// Handle dataset selection for prediction
document.getElementById('predictDataset').addEventListener('change', function() {
    const dataset = this.value;
    if (!dataset) return;
    
    const modelSelect = document.getElementById('predictModel');
    
    // Clear existing options
    modelSelect.innerHTML = '<option value="">Select Model</option>';
    
    // Fetch available models for the selected dataset
    fetch(`/get_saved_models?dataset=${dataset}`)
        .then(response => response.json())
        .then(data => {
            if (data.models.length === 0) {
                const option = document.createElement('option');
                option.value = '';
                option.textContent = 'No models available';
                option.disabled = true;
                modelSelect.appendChild(option);
            } else {
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = `saved_models/${dataset}/${model}`;
                    option.textContent = model;
                    modelSelect.appendChild(option);
                });
            }
        })
        .catch(error => {
            console.error('Error fetching models:', error);
            showToast('Error loading models', 'error');
        });
});

// Handle training form submission
document.getElementById('trainForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = {
        dataset: document.getElementById('trainDataset').value,
        model: document.getElementById('trainModel').value,
        batch_size: document.getElementById('batchSize').value,
        num_epochs: document.getElementById('epochs').value,
        learning_rate: document.getElementById('learningRate').value
    };
    
    // Reset accuracy chart data
    trainAccData = [];
    testAccData = [];
    epochLabels = [];
    if (accuracyChart) {
        accuracyChart.destroy();
        accuracyChart = null;
    }
    
    // Show training progress section
    document.getElementById('trainingProgress').classList.remove('d-none');
    // Hide confirm button initially
    document.getElementById('confirmTraining').classList.add('d-none');
    
    // Disable form during training
    const formElements = this.elements;
    for (let i = 0; i < formElements.length; i++) {
        formElements[i].disabled = true;
    }
    
    fetch('/train', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            showToast('训练完成！', 'success');
            // Show confirm button when training is complete
            document.getElementById('confirmTraining').classList.remove('d-none');
        } else {
            showToast(data.message, 'error');
            // Re-enable form on error
            for (let i = 0; i < formElements.length; i++) {
                formElements[i].disabled = false;
            }
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showToast('训练过程中发生错误', 'error');
        // Re-enable form on error
        for (let i = 0; i < formElements.length; i++) {
            formElements[i].disabled = false;
        }
    });
});

// Handle confirm button click
document.getElementById('confirmTraining').addEventListener('click', function() {
    // Hide training progress section
    document.getElementById('trainingProgress').classList.add('d-none');
    // Hide confirm button
    this.classList.add('d-none');
    // Re-enable form
    const formElements = document.getElementById('trainForm').elements;
    for (let i = 0; i < formElements.length; i++) {
        formElements[i].disabled = false;
    }
    // Reset progress bar
    const progressBar = document.querySelector('#trainingProgress .progress-bar');
    progressBar.style.width = '0%';
    progressBar.setAttribute('aria-valuenow', 0);
    // Reset text information
    document.getElementById('currentEpoch').textContent = 'Epoch: 0/0';
    document.getElementById('currentAccuracy').textContent = '准确率: 0%';
    document.getElementById('currentLR').textContent = '0.001';
    // Reset accuracy chart
    if (accuracyChart) {
        accuracyChart.destroy();
        accuracyChart = null;
    }
    trainAccData = [];
    testAccData = [];
    epochLabels = [];
});

// Handle prediction form submission
document.getElementById('predictForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    
    // Disable form during prediction
    const formElements = this.elements;
    for (let i = 0; i < formElements.length; i++) {
        formElements[i].disabled = true;
    }
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            // Show prediction results
            document.getElementById('predictionResults').classList.remove('d-none');
            
            // Update prediction image
            document.getElementById('predictionImage').src = data.image_path;
            
            // Update prediction list
            const predictionList = document.getElementById('predictionList');
            predictionList.innerHTML = '';
            
            data.predictions.forEach((pred, index) => {
                const item = document.createElement('div');
                item.className = 'list-group-item';
                item.innerHTML = `
                    <div class="d-flex justify-content-between align-items-center">
                        <span>${index + 1}. ${pred[0]}</span>
                        <span class="badge bg-primary rounded-pill">${(pred[1] * 100).toFixed(2)}%</span>
                    </div>
                `;
                predictionList.appendChild(item);
            });
            
            showToast('Prediction completed successfully!', 'success');
        } else {
            showToast(data.message, 'error');
            // Re-enable form on error
            for (let i = 0; i < formElements.length; i++) {
                formElements[i].disabled = false;
            }
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showToast('An error occurred during prediction', 'error');
        // Re-enable form on error
        for (let i = 0; i < formElements.length; i++) {
            formElements[i].disabled = false;
        }
    });
});

// Handle prediction confirm button click
document.getElementById('confirmPrediction').addEventListener('click', function() {
    // Hide prediction results
    document.getElementById('predictionResults').classList.add('d-none');
    // Re-enable form
    const formElements = document.getElementById('predictForm').elements;
    for (let i = 0; i < formElements.length; i++) {
        formElements[i].disabled = false;
    }
    // Clear file input
    document.getElementById('imageFile').value = '';
}); 