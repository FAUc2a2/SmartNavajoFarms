async function accessFRD() {
    try {
        const response = await fetch("https://prod-51.eastus.logic.azure.com/workflows/aa49877dfabc43efbe5eec9c61f2980b/triggers/When_a_HTTP_request_is_received/paths/invoke?api-version=2016-10-01&sp=%2Ftriggers%2FWhen_a_HTTP_request_is_received%2Frun&sv=1.0&sig=2v3EkuvhXwGfalu901LOgpU48S8kJ6Rn8aNqPJFzNAU&database=sensors", {
            method: "GET",

        });;
        if (!response.ok) {
            throw new Error(`Response status: ${response.status}`);
        }

        const json = response.json();


        return json
    } catch (error) {
        throw new Error('Error accessing FRD!');
    }
}

