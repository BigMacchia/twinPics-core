use assert_cmd::Command;

#[test]
fn twinpics_cli_help_exits_zero() {
    let mut cmd = Command::cargo_bin("twinpics_cli").expect("twinpics_cli binary");
    cmd.arg("--help").assert().success();
}
