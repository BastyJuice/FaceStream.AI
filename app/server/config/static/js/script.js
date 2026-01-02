const colorPicker = document.getElementById('colorPicker');
const transparencySlider = document.getElementById('overlayTransparency');
const colorOverlay = document.getElementById('colorOverlay');

// Format timestamps for display.
// The backend currently stores timestamps as *local time strings* (e.g. "2026-01-01 18:00:00").
// Treat those as local time (do NOT append "Z"), otherwise the browser will shift the time.
// If the string already contains a timezone ("Z" or +hh:mm), we let the browser parse it normally.
function formatLocalTime(value) {
    if (!value) return "";

    // If numeric epoch (seconds or ms)
    if (typeof value === 'number') {
        const ms = value < 1e12 ? value * 1000 : value;
        return new Date(ms).toLocaleString();
    }

    const s = String(value).trim();

    // ISO strings with timezone info -> parse directly
    if (/Z$/.test(s) || /[+-]\d\d:?\d\d$/.test(s)) {
        const d = new Date(s);
        return isNaN(d.getTime()) ? s : d.toLocaleString();
    }

    // "YYYY-MM-DD HH:MM:SS" (local time)
    const m = s.match(/^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})(?::(\d{2}))?$/);
    if (m) {
        const year = parseInt(m[1], 10);
        const month = parseInt(m[2], 10) - 1;
        const day = parseInt(m[3], 10);
        const hour = parseInt(m[4], 10);
        const minute = parseInt(m[5], 10);
        const second = parseInt(m[6] || '0', 10);
        const d = new Date(year, month, day, hour, minute, second);
        return isNaN(d.getTime()) ? s : d.toLocaleString();
    }

    // Fallback: try browser parsing
    const d = new Date(s);
    return isNaN(d.getTime()) ? s : d.toLocaleString();
}


function openModalAndShowImages(data) {
    var imageBlock = `<div class="event-image"><img src="${data.image_path}" class="d-block w-100"></div>`;
    var e = document.querySelector('.image-wrapper');
    e.innerHTML = imageBlock;
    var imageModalLabel = document.querySelector('#imageModalLabel')
    imageModalLabel.innerHTML = `${data.name} - ${formatLocalTime(data.timestamp)}`

    var modalElement = document.getElementById('imageModal');
    var modal = new bootstrap.Modal(modalElement);
    modal.show();


}

var table = new Tabulator("#eventlog-table", {
        height: '600px',
        layout: 'fitColumns',
        columns: [
            {title: "Name", field: "name", sorter: "string", width: 200},
            {
                title: "Time",
                field: "timestamp",
                formatter: function (cell, formatterParams) {
                    let value = cell.getValue();
                    return formatLocalTime(value);  // Nutze die Funktion, um das Datum zu formatieren

                },
            },
            {title: "Image Path", field: "image_path"},
        ],

    })
;


function sendBaseUrlToServer() {
    const baseUrl = window.location.origin;
    const requestOptions = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({baseUrl})
    };

    fetch('/api/setBaseUrl', requestOptions)
        .then(response => response.json())
        .catch(error => console.error('Error:', error));
}


function updateOverlay() {
    const color = colorPicker.value;
    const transparency = transparencySlider.value / 100;
    colorOverlay.style.backgroundColor = `rgba(${parseInt(color.substr(1, 2), 16)}, ${parseInt(color.substr(3, 2), 16)}, ${parseInt(color.substr(5, 2), 16)}, ${1 - transparency})`;
}

colorPicker.addEventListener('input', updateOverlay);
transparencySlider.addEventListener('input', updateOverlay);

function updateFacesList() {
    fetch('/list-faces')
        .then(response => response.text())  // Die Antwort als Text verarbeiten
        .then(html => {
            // Ersetzen Sie den Inhalt der Gesichterliste mit dem neuen HTML
            document.querySelector('#personsContainer').innerHTML = html;
            // Re-init dynamic Dropzones after replacing HTML
            if (typeof window.initPersonDropzones === 'function') {
                window.initPersonDropzones();
            }
        })
        .catch(error => {
            console.error('Fehler beim Aktualisieren der Gesichterliste:', error);
        });
}

// Dynamic Dropzones for each person (must be global, because updateFacesList() is global)
if (window.Dropzone) {
    Dropzone.autoDiscover = false;
}

window.personDropzones = window.personDropzones || [];

window.initPersonDropzones = function initPersonDropzones() {
    if (!window.Dropzone) return;

    // Destroy existing instances (avoid duplicate bindings after ajax reload)
    (window.personDropzones || []).forEach(dz => {
        try { dz.destroy(); } catch (e) {}
    });
    window.personDropzones = [];

    document.querySelectorAll('form.person-dropzone').forEach(form => {
        const person = form.dataset.person || '';
        const dz = new Dropzone(form, {
            url: '/upload_faces',
            acceptedFiles: "image/jpeg,image/png,image/jpg",
            maxFilesize: 12,
            dictInvalidFileType: "Ungültiges Dateiformat. Nur JPEG und PNG sind erlaubt."
        });

        dz.on('sending', function (file, xhr, formData) {
            formData.append('person', person);
        });

        dz.on('success', function (file) {
            try { this.removeFile(file); } catch (e) {}
            updateFacesList();
        });

        dz.on('error', function (file, message, xhr) {
            // If backend returns JSON {error: ...}, show that.
            try {
                if (xhr && xhr.responseText) {
                    const data = JSON.parse(xhr.responseText);
                    if (data && data.error) message = data.error;
                }
            } catch (e) { /* ignore */ }

            // Remove the preview immediately so user can upload again without refresh
            try { this.removeFile(file); } catch (e) {}

            alert(message);

            // Refresh list to ensure nothing "sticks" in the UI
            updateFacesList();
        });

        window.personDropzones.push(dz);
    });
};

function updateDelayDisplay(value) {
    const minutes = Math.floor(value / 60);
    const seconds = value % 60;
    document.getElementById('notificationDelay').innerText = minutes > 0 ? `${minutes} minute(s) ${seconds} second(s)` : `${value} second(s)`;
}

function setSize(width, height) {
    document.getElementById('outputWidth').value = width;
    document.getElementById('outputHeight').value = height;
}

function updateFaceRecognitionIntervalValue(value) {
    document.getElementById('faceRecognitionIntervalValue').innerHTML = value;
}

document.addEventListener("DOMContentLoaded", function () {

    // "Open Stream" button (always points to http://<current-host>:5001/stream)
    const openStreamButton = document.getElementById('openStreamButton');
    if (openStreamButton) {
        const host = window.location.hostname;
        openStreamButton.href = `http://${host}:5001/stream`;
    }

    sendBaseUrlToServer();
    function loadEventLog() {
        // Cache-bust to avoid stale Event Log data without requiring a hard refresh.
        table.setData(`/event_log?ts=${Date.now()}`);
    }

    loadEventLog();
    table.on("rowClick", function (e, row) {
        var data = row.getData();
        openModalAndShowImages(data);
    });


    // Reload event log whenever the user opens the Eventlog tab.
    const eventlogTab = document.getElementById('eventlog-tab');
    if (eventlogTab) {
        eventlogTab.addEventListener('shown.bs.tab', function () {
            loadEventLog();
        });
    }

//form submit
    document.getElementById('submitFormButton').addEventListener('click', async function (event) {
        event.preventDefault(); // Verhindere das normale Abschicken des Formulars
        let self = this;

        let settingsForm = document.getElementById('settings-form');
        let notificationForm = document.getElementById('notification-form');

        if (!settingsForm.checkValidity() || !notificationForm.checkValidity()) {
            settingsForm.classList.add('was-validated');
            notificationForm.classList.add('was-validated');
            return; // Stoppe die Ausführung, wenn die Formulare ungültig sind
        }

        let formData = new FormData(settingsForm);
        new FormData(notificationForm).forEach((value, key) => formData.append(key, value));

        this.disabled = true; // Deaktiviere den Button
        this.classList.add('disabled');

        try {
            let response = await fetch(settingsForm.action, {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                // Zeige das Modal an
                let modal = new bootstrap.Modal(document.getElementById('successModal'));
                modal.show();

                // Starte den Countdown
                let countdownElement = document.getElementById('countdown');
                let timeLeft = 20; // Zeit in Sekunden

                let timerId = setInterval(() => {
                    timeLeft--;
                    countdownElement.textContent = timeLeft;
                    if (timeLeft <= 0) {
                        clearInterval(timerId);
                        window.location.reload(); // Seite neu laden
                    }
                }, 1000);
            } else {
                throw new Error('Server antwortete mit einem Fehler: ' + response.status);
            }
        } catch (error) {
            console.error('Fehler beim Senden der Formulardaten', error);
            alert('Ein Fehler ist aufgetreten: ' + error.message);
        } finally {
            setTimeout(() => {
                self.disabled = false;
                self.classList.remove('disabled');
            }, 2000); // Wieder aktivieren nach 2 Sekunden
        }
    });


    document.getElementById('overlayTransparency').oninput = function () {
        document.getElementById('transparencyValue').innerText = this.value + '%';
    };

    // Init dynamic Dropzones (global helper, used by updateFacesList())
    if (typeof window.initPersonDropzones === 'function') {
        window.initPersonDropzones();
    }

    // Create person
    const createBtn = document.getElementById('createPersonBtn');
    if (createBtn) {
        createBtn.addEventListener('click', function () {
            const input = document.getElementById('newPersonName');
            const person = (input ? input.value : '').trim();
            if (!person) return;

            fetch('/create_person', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({person})
            })
                .then(res => res.json())
                .then(() => {
                    if (input) input.value = '';
                    updateFacesList();
                })
                .catch(err => console.error('Fehler beim Anlegen der Person:', err));
        });
    }

    // (Dropzones already initialized above)


    // Intercept delete actions (keep current tab, no full page reload)
    const personsContainer = document.getElementById('personsContainer');
    if (personsContainer) {
        personsContainer.addEventListener('submit', async (ev) => {
            const form = ev.target;
            if (!(form instanceof HTMLFormElement)) return;

            if (form.classList.contains('js-delete-image')) {
                ev.preventDefault();
                if (!confirm('Delete this image?')) return;
                try {
                    const resp = await fetch(form.action, { method: 'POST' });
                    if (!resp.ok) throw new Error('Delete failed');
                    await updateFacesList();
                } catch (e) {
                    console.error(e);
                    alert('Failed to delete image.');
                }
            }

            if (form.classList.contains('js-delete-person')) {
                ev.preventDefault();
                if (!confirm('Delete this person and all images?')) return;
                try {
                    const resp = await fetch(form.action, { method: 'POST' });
                    if (!resp.ok) throw new Error('Delete failed');
                    await updateFacesList();
                } catch (e) {
                    console.error(e);
                    alert('Failed to delete person.');
                }
            }
        });
    }

});
