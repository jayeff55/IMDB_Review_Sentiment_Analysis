import Dropdown from 'react-bootstrap/Dropdown';

function PageWelcome() {
    return (
        <div>
            <h1>Sentiment Analyzer</h1>
            <p>Want to put this sentiment analyzer to the test? Try it out by leaving a movie review and hit OK! You get to pick from 5 different analyzers</p>
        </div>
    )
}

function ModelDropdown() {
    return (
        <Dropdown>
            <Dropdown.Toggle variant="success" id="dropdown-basics">
                Choose a model
            </Dropdown.Toggle>

            <Dropdown.Menu>
                {["model 1", "model 2", "model 3"].map(
                    (model) => (
                        <Dropdown.Item>{model}</Dropdown.Item> //add href to dropdown.Item prop
                    )
                )}
            </Dropdown.Menu>
        </Dropdown>

    )
}

function InputTextbox() {
    return(
        <form action="/submit" method="post">
            <input type="text" name="inputText"></input>
            <button type="submit">Submit my review</button>
        </form>
    )
}