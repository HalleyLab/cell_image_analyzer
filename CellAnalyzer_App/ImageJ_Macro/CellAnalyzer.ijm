// Fiji/ImageJ macro bridge for the Python cell-analysis pipeline.
// Save a complete session YAML from the desktop app before running this macro.
requires("1.52u");

project = getDirectory("Choose the cell_analyzer project folder");
if (project == "") exit("No project folder selected.");
session = File.openDialog("Choose cell_analysis_session.yaml");
if (session == "") exit("No session selected.");

Dialog.create("Cell Analyzer");
Dialog.addString("Python", "C:\\Program Files\\Python312\\python.exe", 55);
Dialog.addCheckbox("Wait until analysis finishes", true);
Dialog.show();
python = Dialog.getString();
waitForCompletion = Dialog.getCheckbox();

runner = project + "run_cell_analysis_cli.py";
if (!File.exists(runner)) exit("Cannot find: " + runner);
if (!waitForCompletion) setOption("WaitForCompletion", false);

showStatus("Running cell analysis...");
output = exec(python, runner, "session", "--session", session);
if (waitForCompletion) {
    print("Cell Analyzer output:\n" + output);
    showMessage("Cell Analyzer", "Analysis finished.\nSee the Log window and the selected output folder.");
}
