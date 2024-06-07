// Code existant pour la soumission du formulaire
document.getElementById("Form").addEventListener("submit", function(event) {
    var selectedValue = document.getElementById("dropdown").value;
    if (selectedValue !== "") {
        // Prevent default form submission
        event.preventDefault();
        
        // Update the form action based on the selected value
        this.action = "/showResult/";
        
        // Submit the form programmatically
        this.submit();
    }
});