// Helper to create typed context
import { createContext, FC, ReactNode, useContext } from 'react';

export const makeContext = <T,>(render: () => T) => {
    const Context = createContext<T>({} as T);

    const useContextHook = () => useContext(Context);

    const Provider: FC<{ children: ReactNode }> = ({ children }) => {
        const value = render();
        return <Context.Provider value={value}>{children}</Context.Provider>;
    };

    return [Provider, useContextHook] as const;
};
