import simpy
class   C_Spec () :
    def __init__ (self, env, name, nID, fBuildingCT, fCuringCT, nInitQuantity) :
        self.m_env = env

        self.m_name = name
        self.m_nID = nID
        self.m_fBuildingCT = fBuildingCT
        self.m_fCuringCT = fCuringCT

        self.m_store = simpy.Store(self.m_env)
        self.m_nLastSeq = 0

        self.reset_store (nInitQuantity)

    def get_lastSeq (self) :
        return self.m_nLastSeq

    def reset_store (self, nInitQuantity) :
        for i in range (nInitQuantity) :
            self.m_env.process(self.store_put_proc())

    def store_put_proc (self) :
        self.m_nLastSeq += 1
        nLastSeq = self.m_nLastSeq
        yield self.m_store.put(f'{self.m_name}{nLastSeq}')  # 넣어주는 동시에 다른 yield get에서 item을 가져간다.
        print (f'put {self.m_name}{nLastSeq}')