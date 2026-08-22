from ncert_rag import main


def test_main_prints(capsys):
    main.main()
    assert capsys.readouterr().out == "Hello World\n"
