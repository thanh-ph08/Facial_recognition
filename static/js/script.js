const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startWebcamBtn = document.getElementById('startWebcamBtn');

let stream = null;

const showAllBtn = document.getElementById('show-all-btn');
const showPresentBtn = document.getElementById('show-present-btn');
const showAbsentBtn = document.getElementById('show-absent-btn');
const studentListDiv = document.getElementById('student-list');

const searchInput = document.getElementById('search-input');

const autoRecBtn = document.getElementById('autoRecBtn');

let streaming = false;
let streamInterval = null;

let currentFilter = 'all';
let currentSearch = '';

function renderStudents(filter = 'all', search = '') {
    fetch('/students')
        .then(res => res.json())
        .then(data => {
            studentListDiv.innerHTML = '';
            data.forEach((s, i) => {
                const matchFilter =
                    filter === 'all' ||
                    (filter === 'present' && s.present) ||
                    (filter === 'absent' && !s.present);
                const matchSearch = s.name.toLowerCase().includes(search.toLowerCase());
                if (matchFilter && matchSearch) {
                    const div = document.createElement('div');
                    div.className = 'student-box' + (s.present ? ' present' : ' absent');
                    div.id = `student-${i+1}`;
                    div.textContent = s.name;
                    studentListDiv.appendChild(div);
                }
            });
        });
}

showAllBtn.addEventListener('click', () => {
    setActiveBtn(showAllBtn);
    currentFilter = 'all';
    renderStudents(currentFilter, currentSearch);
});
showPresentBtn.addEventListener('click', () => {
    setActiveBtn(showPresentBtn);
    currentFilter = 'present';
    renderStudents(currentFilter, currentSearch);
});
showAbsentBtn.addEventListener('click', () => {
    setActiveBtn(showAbsentBtn);
    currentFilter = 'absent';
    renderStudents(currentFilter, currentSearch);
});

searchInput.addEventListener('input', () => {
    currentSearch = searchInput.value;
    renderStudents(currentFilter, currentSearch);
});

renderStudents('all', '');

// Đánh dấu nút đang chọn
function setActiveBtn(btn) {
    [showAllBtn, showPresentBtn, showAbsentBtn].forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
}

// Bật webcam khi nhấn nút
startWebcamBtn.addEventListener('click', () => {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(s => {
            stream = s;
            video.srcObject = stream;
            autoRecBtn.disabled = false; // Bật nút nhận diện liên tục khi webcam đã bật
        })
        .catch(err => {
            console.error("Lỗi khi truy cập camera: ", err);
            alert("Không thể truy cập camera. Vui lòng kiểm tra quyền truy cập.");
        });
});



function markStudentPresent(name) {
    fetch('/mark_present', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name})
    }).then(() => {
        renderStudents(currentFilter, currentSearch);
    });
}



function sendFrameToBackend() {
    if (!streaming) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/jpeg', 0.7);
    fetch('/stream_recognize', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({image: dataURL})
    }).then(() => {
        // Cập nhật lại danh sách ngay sau mỗi lần nhận diện
        renderStudents(currentFilter, currentSearch);
        // Gửi frame tiếp theo sau 500ms
        if (streaming) setTimeout(sendFrameToBackend, 500);
    });
}


document.querySelector('.button-container').appendChild(autoRecBtn);

autoRecBtn.addEventListener('click', () => {
    streaming = !streaming;
    autoRecBtn.textContent = streaming ? "Dừng nhận diện" : "Nhận diện";
    if (streaming) sendFrameToBackend();
});
