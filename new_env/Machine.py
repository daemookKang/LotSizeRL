from collections import deque
import simpy

class   C_Machine () :
    def __init__(self, env, name):
        self.m_env = env
        self.m_name = name

        self.m_bStatus = False
        self.m_mfg_proc = None
        self.m_watchdog_proc = None

        self.m_curSpec = None
        self.m_nTotal = 0
        self.m_nCurProd = 0
        self.m_nUntil = 86400

        self.m_queue = deque()

    def is_busy (self) :
        return self.m_bStatus

    def add_queue (self, spec, nTotal) :
        self.m_queue.appendleft ((spec, nTotal))

    def set_sim_until (self, nUntil) :
        self.m_nUntil = nUntil


    def start_watchdog_proc (self) :
        self.m_watchdog_proc = self.m_env.process(self.watchdog_proc_generator())

    def start_mfg_proc (self):
        self.m_mfg_proc = self.m_env.process(self.mfg_proc_generator())

    def watchdog_proc_generator (self):
        raise NotImplementedError()

    def mfg_proc_generator (self):
        raise NotImplementedError()





class   C_CuringMch (C_Machine) :
    def __init__ (self, env, name):
        super().__init__(env, name)
        self.m_nBackOrderTime = 0
        self.m_backOrder_lst = []

        self.m_nTimeLimit = 86400
        self.m_nReqTime = 86400 # Curing에서 계속 요청하는데 창고가 막혀있으면 사용한다.
        self.m_bReq = False

        self.start_mfg_proc()
        self.start_watchdog_proc()


    def watchdog_proc_generator (self):
        while (self.m_env.now < self.m_nUntil-1) :
            try :
                yield self.m_env.timeout(1) # 86399에서 1을 delay하는 바람에 + 1이 올라갔다.
                if (self.m_bReq) :
                    if (self.m_env.now - self.m_nReqTime > self.m_nTimeLimit) :
                        print (f'{self.m_env.now}')
                        self.m_mfg_proc.interrupt()
                        print ('Interrupt1')

            except :
                print(f'{self.m_curSpec.m_name}{self.m_env.now}')
                print('restart')
                self.start_mfg_proc()




    def mfg_proc_generator (self) :
        while (self.m_env.now < self.m_nUntil-1):
            if (self.m_bStatus) :
                self.m_nCurProd = 0
                try:
                    for i in range(self.m_nTotal):
                        t1 = self.m_env.now
                        self.m_nReqTime = t1

                        self.m_bReq = True
                        item = yield self.m_curSpec.m_store.get()
                        self.m_bReq = False

                        t2 = self.m_env.now
                        if (t2 > t1):
                            print(f'delay {t2 - t1}')
                            self.m_nBackOrderTime += (t2 - t1)
                            self.m_backOrder_lst.append ((item, self.m_nBackOrderTime, t2))
                        else :
                            self.m_nBackOrderTime = 0
                            #yield self.m_env.timeout(self.m_curSpec.m_fCuringCT) # 바로 뽑아오는 경우에는 curing 되는 사이에 오니까 운반 시간 추가 필요 X

                        print(f'get {item} from {self.m_name} at {self.m_env.now}')
                        yield self.m_env.timeout(self.m_curSpec.m_fCuringCT)
                        self.m_nCurProd += 1

                    self.m_bStatus = False
                    self.m_nTotal = 0
                except simpy.Interrupt:
                    print('Interrupt2')
                    return None

            else :
                if (len(self.m_queue) > 0):  # 처리할 작업이 있는 경우에만 처리한다.
                    self.m_curSpec, self.m_nTotal = self.m_queue.pop()
                    self.m_bStatus = True

                else :
                    yield self.m_env.timeout(1)


class   C_BuildingMch (C_Machine) :
    def __init__ (self, env, name):
        super().__init__(env, name)

        self.start_mfg_proc()
        self.start_watchdog_proc()

    def watchdog_proc_generator (self):
        while (self.m_env.now < self.m_nUntil-1) :
            yield self.m_env.timeout (1)

    def mfg_proc_generator(self):
        while (self.m_env.now < self.m_nUntil-1):
            if (self.m_bStatus):
                self.m_nCurProd = 0
                for i in range(self.m_nTotal):
                    id = len(self.m_curSpec.m_store.items) + 1

                    yield self.m_env.timeout(self.m_curSpec.m_fBuildingCT)
                    self.m_curSpec.m_nLastSeq += 1
                    yield self.m_curSpec.m_store.put(f'{self.m_curSpec.m_name}{self.m_curSpec.get_lastSeq ()}')
                    #yield self.m_curSpec.store_put_proc ()
                    # 창고에 입고되는 시간 yield self.m_env.timeout(100)
                    print(f'put {self.m_curSpec.m_name}{self.m_curSpec.get_lastSeq ()} at {self.m_env.now}')

                    self.m_nCurProd += 1

                self.m_bStatus = False
                self.m_nTotal = 0

            else:
                if (len(self.m_queue) > 0):  # 처리할 작업이 있는 경우에만 처리한다.
                    demand_tuple = self.m_queue.pop()
                    self.m_curSpec = demand_tuple[0]
                    self.m_nTotal = demand_tuple[1]
                    self.m_bStatus = True
                ####### 여기서 decision making을 해도 된다. #####

            yield self.m_env.timeout(1)



