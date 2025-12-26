// UI
    const toggleBtn = document.getElementById('toggleBtn');
    const sidebar = document.getElementById('sidebar');
    const main = document.getElementById('main');

    toggleBtn.addEventListener('click', () => {
      sidebar.classList.toggle('hidden');

      if (window.innerWidth > 768) {
        main.classList.toggle('full');
      }
    });

    document.addEventListener('click', (e) => {
      if (
        window.innerWidth <= 768 &&
        !sidebar.contains(e.target) &&
        !toggleBtn.contains(e.target)
      ) {
        sidebar.classList.add('hidden');
      }
    });
// UI


document.getElementById("startBtn").addEventListener("click",()=>{
  fetch("/start_tracking",{method:"POST"})
  .then(()=>{
    console.log("Tracking enabled");
  })
  .catch((err)=> console.error(err))
});

document.getElementById("stopBtn").addEventListener("click",()=>{
  fetch("/stop_tracking",{method:"POST"})
  .then(()=>{
    console.log("Tracking disabled");
  })
  .catch((err)=> console.error(err))
});

document.getElementById("sidebar").addEventListener("click", function(e) {
if (e.target.classList.contains("sidebar-item")) {
    const selectedExercise = e.target.getAttribute("exercise");
    
    document.querySelectorAll('.sidebar-item').forEach(item => {
    item.classList.remove('color-gradient-bg');
    });    
    e.target.classList.add('color-gradient-bg');
    document.getElementById("exercise-select").textContent = e.target.textContent;


    const singleRepContainer = document.querySelector('.single-rep-container');
    const doubleRepContainer = document.querySelector('.double-rep-container');

    if (selectedExercise === 'squat' || selectedExercise === 'pushup') {
        singleRepContainer.style.display = 'block';
        doubleRepContainer.style.display = 'none';
    } else {
        singleRepContainer.style.display = 'none';
        doubleRepContainer.style.display = 'flex';
    }

    fetch("/select_exercise", {
        method: "POST",
        body: new URLSearchParams({ exercise: selectedExercise }),
        headers: { "Content-Type": "application/x-www-form-urlencoded" }
    }).then(() => {
        document.getElementById("video-feed").src = "/video_feed?" + new Date().getTime();
    });
}
});




function updateReps() {
    fetch('/get_reps')
        .then(response => response.json())
        .then(data => {
            if (Array.isArray(data)) {
                const leftData = data[0];
                const rightData = data[1];

                // Update double-rep view
                document.getElementById('left-rep-count').textContent = leftData.left_counter;
                document.getElementById('right-rep-count').textContent = rightData.right_counter;
                document.getElementById('left-stage').textContent = leftData.left_stage || '-';
                document.getElementById('right-stage').textContent = rightData.right_stage || '-';
            }

            else if (data && typeof data === 'object') {
                document.getElementById('rep-count').textContent = data.left_counter;
                document.getElementById('stage').textContent = data.left_stage || '-';

            }
        })
        .catch(err => console.error("Fetch error in updateReps:", err));
}



setInterval(updateReps, 140); // Update every 140 milliseconds
