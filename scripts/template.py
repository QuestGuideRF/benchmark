"""
@Desc: модуль шаблона
@Файл: template.py
@Автор: ляньцзиньхуа
@Дата: 30.05.2019, 19:26
"""
import os

MAIL_HEAD_CONTENT ="""От:paddle_benchmark@baidu.com
Кому: test@benchmark.com
Тема:【DEVICE_HOLDER_GRAPH_HOLDER_JOB_HOLDER_CUDA_HOLDER-TIME_HOLDER】运行结果报警,请检查
тип контента: текст/html
<html>
    <тело>
        <h3 align=center>Письмо о тревоге TITLE_HOLDER</h3>
        <table style="display:DISPLAY;" граница="1" выравнивание=центр>
        <caption bgcolor="#989898">Цветные изображения</caption>
FAIL_JOB_HOLDER
        </таблица>"""

MAIL_MID_CONTENT ="""<HR align=center width="80%" SIZE=1>
        <table border="1" align=center>
        <caption>报警结果_INDEX_HOLDER</caption>
            <tr bgcolor="#989898" >
TABLE_HEADER_HOLDER
            </tr>
ALARM_INFO_HOLDER
        </таблица>"""

MAIL_TAIL_CONTENT ="""<HR align=center width="80%" SIZE=1>
        <table border="1" align=center>
        <caption bgcolor="#989898">Цвета</caption>
RUN_ENV_HOLDER
        </таблица>
        <HR align=center width="80%" SIZE=1>
        <div style="text-align:center;">
        <a align=center href="BENCHMARK_WEBSITE1">参考链接</a><br/><br/>
        <a align=center href="BENCHMARK_WEBSITE2">Эталонный тест</a><br/><br/>
        </div>
    </тело>
</html>"""

class EmailTemplate (object ):
    """создать электронное письмо для получения результатов теста."""

    def __init__ (self ,title ,env ,results ,log_path ,fail_jobs =[]):
        """        Аргументы:
                    заголовок(строка): эталон | оп_бенчмарк | сравнительное_распределение
                    env(dict): рабочая среда.
                    результаты (дикт):
                        {"Скорость": {"header": [table_header0, table_header1, table_header2,]
                                   "данные": [[{'значение':, 'цвет':, }, {'значение':, 'цвет':, }, {'значение':, 'цвет':, }]
                                   ...]}
                        "Mem": {"header": [table_header0, table_header1, table_header2,]
                                "данные": [[{'значение':, 'цвет':, }, {'значение':, 'цвет':, }, {'значение':, 'цвет':, }]
                                ...]}
                        ...}
                    log_path(str): путь к почте"""
        self .title =title
        self .fail_jobs =fail_jobs
        self .job_display ='none'
        self .fail_job_content =""
        self .env_content =""
        self .alarm_info =""
        self .log_path =log_path
        self .__construct_mail_env (env )
        self .__construct_alarm_info (results )
        self .__construct_fail_job (fail_jobs )

    def __construct_mail_env (self ,env ):
        """        создать содержимое окружения электронной почты."""

        if isinstance (env ,dict ):
            for k ,v in env .items ():
                self .env_content +="""
                    <tr><td>{}</td><td>{}</td></tr>
                    """.format (k ,v )
        return self .env_content

    def __construct_fail_job (self ,fail_jobs ):
        """        создать содержимое неудачного задания."""
        if fail_jobs :
            self .job_display ='block'
            for job in fail_jobs :
                self .fail_job_content +="""
                     <tr><td>{}</td><td>{}</td></tr>
                     """.format (job [0 ],job [1 ])
        else :
            self .job_display ='none'
        return self .fail_job_content

    def __construct_alarm_info (self ,results ):
        """        создать содержимое окружения электронной почты."""
        print ("html_results: {}".format (results ))
        for report_index ,alarm in results .items ():
            table_header_info =""
            table_alarm_info =""
            if not alarm .get ("data"):
                continue
            for header_td in alarm ["header"]:
                table_header_info +="""
                    <td>{}</td>
                    """.format (header_td )
            for single_info in alarm ["data"]:
                if not single_info :
                    continue
                table_alarm_info +="\t\t\t<tr>"
                for info_td in single_info :
                    table_alarm_info +="""
                    <td bgcolor={}>{}</td>
                    """.format (info_td .get ('color','white'),info_td .get ('value'))
                table_alarm_info +="</tr>\n"
            if table_alarm_info :
                self .alarm_info +=MAIL_MID_CONTENT .replace ('TABLE_HEADER_HOLDER',table_header_info ).replace (
                'ALARM_INFO_HOLDER',table_alarm_info ).replace ('INDEX_HOLDER',report_index )

    def construct_email_content (self ):
        """        создать полное содержание электронной почты."""

        content =MAIL_HEAD_CONTENT .replace ("TITLE_HOLDER",self .title ).replace ('FAIL_JOB_HOLDER',
        self .fail_job_content ).replace (
        "TIME_HOLDER",os .getenv ("START_TIME")).replace ("GRAPH_HOLDER",os .getenv ("BENCHMARK_GRAPH")).replace (
        "JOB_HOLDER",os .getenv ("BENCHMARK_TYPE")).replace ("DEVICE_HOLDER",os .getenv ("DEVICE_TYPE")).replace ("CUDA_HOLDER",os .getenv ("VERSION_CUDA")).replace ('DISPLAY',self .job_display )

        if not self .alarm_info :
            return

        content +=self .alarm_info

        content +=MAIL_TAIL_CONTENT .replace ("BENCHMARK_WEBSITE1",os .getenv ("BENCHMARK_WEBSITE1","")).strip ().replace (
        'RUN_ENV_HOLDER',self .env_content ).replace ("BENCHMARK_WEBSITE2",os .getenv ("BENCHMARK_WEBSITE2"))

        with open (os .path .join (self .log_path ,"mail.html"),"w")as f_object :
            f_object .write (content )