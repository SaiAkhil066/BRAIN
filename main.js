//REQUEST QUERY
document.getElementById('submitBtn').addEventListener('click', async function() {
    await sendRequest();
});

document.getElementById('userInput').addEventListener('keydown', async function(event) {
    if (event.key === 'Enter') {
        event.preventDefault(); // Prevent the default Enter key behavior (e.g., adding a newline)
        await sendRequest();
    }
});

// ACCORDION UP AND DOWN
document.getElementById('appDescriptionAccordion').addEventListener('shown.bs.collapse', function () {
    document.getElementById('closeAccordionBtn').style.display = 'block';
});

document.getElementById('appDescriptionAccordion').addEventListener('hidden.bs.collapse', function () {
    document.getElementById('closeAccordionBtn').style.display = 'none';
});

function closeAccordion() {
    var accordion = new bootstrap.Collapse(document.getElementById('collapseDescription'), { toggle: false });
    accordion.hide();
}


function displayDocument() {
    // Get the input value
    var fullDocumentName = document.getElementById("documentName").value;
    // Exclude the first 5 characters
    var documentName = fullDocumentName.substring(5);
    // Clear the input field after displaying the PDF
    document.getElementById("documentName").value = "";
    // Assuming the provided path to PDFs
    var pdfPath = `./pdfs/${documentName}.pdf`;

    // Open the PDF in a new window
    var pdfWindow = window.open(pdfPath, '_blank');
}


// TYPE WRITER, PRE-LOADER EFFECTS

// Function to simulate a continuous typewriter effect
async function continuousTypeWriter(element, text, speed) {
    while (true) {
        let i = 0;
        while (i < text.length) {
            element.innerHTML += text.charAt(i);
            i++;
            await new Promise(resolve => setTimeout(resolve, speed));
        }
        await new Promise(resolve => setTimeout(resolve, 1000)); // Pause for 1 second before clearing the text
        element.innerHTML = ''; // Clear the existing text
    }
}

// Simulate a delay for demonstration purposes (you can remove this in your actual code)
setTimeout(function () {
    document.querySelector('.preloader').style.display = 'none';

    // Continuous typewriter effect starts after the preloader is hidden
    const paraElement = document.querySelector('.chat-para');
    const paraText = paraElement.innerText;
    paraElement.innerText = ''; // Clear the existing text
    continuousTypeWriter(paraElement, paraText, 100); // Adjust the speed as needed
}, 2000);

// Update the loading bar width after the preloader is displayed
document.addEventListener('DOMContentLoaded', function () {
    setTimeout(function () {
        document.querySelector('.loading-bar').style.width = '100%';
    }, 100);
});

// Function to send the request
async function sendRequest() {
    // Show loader
    document.getElementById('loader').style.display = 'flex';
    var userInput = document.getElementById('userInput').value;
    if (document.activeElement.id === 'dictionaryBtn'){
        window.location.href = '/dictionary';
        return;
    }
    document.getElementById('response').innerHTML = '<p>Please wait... Artificial Intelligence Initializing...</p>';
    const formData = new FormData();
    formData.append('query', userInput);

    try {
        const response = await fetch('/get_response', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        document.getElementById('response').innerHTML = `<p>${data.answer}</p><br><pre><b>Source Document: </b> ${data.doc}</pre>`;

        // Update the input field with the source document name
        document.getElementById("documentName").value = data.doc;

    } catch (error) {
        console.error('Error:', error);
        document.getElementById('response').innerHTML = '<p>Error processing your request</p>';
    } finally {
        document.getElementById('loader').style.display = 'none';
    }

        // Delay for 5 seconds before clearing the input box
        setTimeout(function() {
            document.getElementById('userInput').value = "";
        }, 5000);


}

let inactivityTimeout;
let forceLogoutTimeout;

function resetInactivityTimeout() {
    // Clear existing timeouts
    clearTimeout(inactivityTimeout);
    clearTimeout(forceLogoutTimeout);

    // Set a new timeout for 5 minutes (300,000 milliseconds)
    inactivityTimeout = setTimeout(function () {
        // Display a warning message using a custom modal
        showWarningModal();

        // Set a new timeout for force logout after 1 minute (60,000 milliseconds)
        forceLogoutTimeout = setTimeout(function () {
            // Perform force logout or cleanup actions here
            // Redirect to logout page or trigger your logout logic
            var confirmationMessage = 'You have been logged out due to inactivity.';
            alert(confirmationMessage);
            window.location.href = '/logout';
        }, 120000); // 1 minute in milliseconds
    }, 300000); // 5 minutes in milliseconds
}

// Event listeners for user interactions
document.addEventListener('mousemove', resetInactivityTimeout);
document.addEventListener('keydown', resetInactivityTimeout);
document.addEventListener('scroll', resetInactivityTimeout);
document.addEventListener('touchstart', resetInactivityTimeout);

// Initial setup
resetInactivityTimeout();

function showWarningModal() {
    // Create a modal element
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-content">
            <p>You have been inactive for 5 minutes. If there is no response, you will be logged out in 1 minute.</p>
        </div>
    `;

    // Append the modal to the document body
    document.body.appendChild(modal);

    // Set a timeout to remove the modal after a few seconds
    setTimeout(function () {
        modal.remove();
    }, 5000); // 5 seconds in milliseconds
}


